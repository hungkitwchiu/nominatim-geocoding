import csv
import requests
from psycopg2 import pool
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from geofunctions import load_cleanup_rules
from geofunctions import expand_abbreviations
from geofunctions import expand_directions
from geofunctions import NAME_CLEANUP_MAP
from geofunctions import FUZZY_SUFFIXES

# Configuration
INPUT_FILE = "CAM_geocoded_pass2_unmatched.csv"
#INPUT_FILE = "CAM_address.csv"
VIEWBOX_FILE = "city_viewboxes.csv"
OUTPUT_FILE_MATCHES = "CAM_geocoded_pass3_matches.csv"
OUTPUT_FILE_UNMATCHED = "CAM_geocoded_pass3_unmatched.csv"
MAX_WORKERS = 10
API_URL = "http://localhost/nominatim/search"

DB_PARAMS = {
    'dbname': 'osm_raw',
    'user': 'nominatim',
    'host': 'localhost',
    'password': 'nominatim'
}

BUFFER_DISTANCE = 1000
connection_pool = None


# --- UTILITY AND QUERY FUNCTIONS ---
def parse_viewbox(viewbox_str):
    parts = list(map(float, viewbox_str.split(',')))
    return parts if len(parts) == 4 else None

def load_viewboxes(file_path):
    viewbox_dict = {}
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            viewbox_str = row['viewbox']
            parsed_coords = parse_viewbox(viewbox_str)
            if parsed_coords is None:
                raise ValueError(f"Malformed viewbox for city '{row['city']}': '{viewbox_str}'")
            viewbox_dict[row['city'].lower()] = parsed_coords
    return viewbox_dict

def extract_city(address):
    parts = address.split(',')
    return parts[-2].strip().lower() if len(parts) >= 2 else None

def try_nominatim(address_to_query, viewbox_coords_list):
    # pass if city column is empty
    if viewbox_coords_list is None: return None, None, "Skipped: No viewbox for Nominatim"
    try:
        left, top, right, bottom = viewbox_coords_list
        viewbox_str_for_api = f"{left},{top},{right},{bottom}"
        response = requests.get(API_URL, params={
            "q": address_to_query, "format": "json", "limit": 1,
            "viewbox": viewbox_str_for_api, "bounded": 1
        })
        response.raise_for_status()
        data = response.json()
        return (float(data[0]["lat"]), float(data[0]["lon"]), "Nominatim match") if data else (None, None, "No Nominatim match")
    except Exception as e:
        return None, None, f"Nominatim error: {e}"

def try_postgis_intersection(street1, street2, db_conn, viewbox_coords):
    """
    Returns (lat, lon, msg) if both features are inside the viewbox,
    or (None, None, reason) otherwise.
    Simplified logic:
      - If both are lines: find exact intersection or fallback to average of centroids.
      - If one is polygon and one is line: take centroids of each and average them.
    """
    if not viewbox_coords:
        return None, None, "Skipped: No viewbox for PostGIS"

    # Unpack viewbox
    min_lon, min_lat, max_lon, max_lat = viewbox_coords
    envelope_sql = "ST_MakeEnvelope(%s, %s, %s, %s, 4326)"
    bbox_params = [min_lon, min_lat, max_lon, max_lat]

    def exists_in(table, name):
        sql = f"""
            SELECT EXISTS(
              SELECT 1 FROM {table}
              WHERE unaccent(name) ILIKE unaccent(%s)
                AND ST_Intersects(way, {envelope_sql})
            );
        """
        params = [f"%{name}%"] + bbox_params
        with db_conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    # Detect feature types
    is_line1 = exists_in("planet_osm_line",    street1)
    is_poly1 = exists_in("planet_osm_polygon", street1)
    is_line2 = exists_in("planet_osm_line",    street2)
    is_poly2 = exists_in("planet_osm_polygon", street2)

    # Check presence
    if not ((is_line1 or is_poly1) and (is_line2 or is_poly2)):
        return None, None, f"PostGIS: '{street1}' or '{street2}' not in viewbox"

    # Helper to get centroid of a table entry
    def get_centroid(table, name):
        if table == 'planet_osm_line':
            sql = f"""
                SELECT ST_Centroid(ST_Union(way)) FROM planet_osm_line
                WHERE unaccent(name) ILIKE unaccent(%s)
                  AND ST_Intersects(way, {envelope_sql});
            """
        else:
            sql = f"""
                SELECT ST_Centroid(way) FROM {table}
                WHERE unaccent(name) ILIKE unaccent(%s)
                  AND ST_Intersects(way, {envelope_sql});
            """
        params = [f"%{name}%"] + bbox_params
        with db_conn.cursor() as cur:
            cur.execute(sql, params)
            geom = cur.fetchone()[0]
        # extract X/Y
        if geom:
            cur = db_conn.cursor()
            cur.execute("SELECT ST_Y(%s), ST_X(%s);", (geom, geom))
            y, x = cur.fetchone()
            return y, x
        return None, None

    # Case A: both lines – try exact intersection first
    if is_line1 and is_line2:
        sql = f"""
            SELECT ST_Y(pt), ST_X(pt) FROM (
              SELECT (ST_Dump(
                        ST_Intersection(l1.way, l2.way)
                      )).geom AS pt
              FROM planet_osm_line l1, planet_osm_line l2
              WHERE unaccent(l1.name) ILIKE unaccent(%s)
                AND unaccent(l2.name) ILIKE unaccent(%s)
                AND ST_Intersects(l1.way, l2.way)
                AND ST_Intersects(l1.way, {envelope_sql})
            ) sub
            WHERE GeometryType(pt)='POINT'
            LIMIT 1;
        """
        params = [f"%{street1}%", f"%{street2}%"] + bbox_params
        with db_conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        if row:
            return row[0], row[1], f"PostGIS: Intersection {street1} & {street2}"
        # fallback: average of centroids
        cy1, cx1 = get_centroid('planet_osm_line', street1)
        cy2, cx2 = get_centroid('planet_osm_line', street2)
        if cy1 and cy2:
            return ( (cy1+cy2)/2, (cx1+cx2)/2,
                     f"PostGIS: Avg centroid {street1} & {street2}" )

    # Case B: one polygon, one line – average their centroids
    if (is_poly1 and is_line2) or (is_line1 and is_poly2):
        p_name, l_name = (street1, street2) if is_poly1 else (street2, street1)
        cy_p, cx_p = get_centroid('planet_osm_polygon', p_name)
        cy_l, cx_l = get_centroid('planet_osm_line',    l_name)
        if cy_p and cy_l:
            return ( (cy_p+cy_l)/2, (cx_p+cx_l)/2,
                     f"PostGIS: Avg centroid {p_name} & {l_name}" )

    # Case C: only one line present – centroid
    if is_line1:
        cy, cx = get_centroid('planet_osm_line', street1)
        if cy:
            return cy, cx, f"PostGIS: Centroid {street1}"

    # Case D: only one polygon present – centroid
    if is_poly1:
        cy, cx = get_centroid('planet_osm_polygon', street1)
        if cy:
            return cy, cx, f"PostGIS: Centroid {street1}"

    return None, None, "No PostGIS match"


def _query_geocoders(address_variant, viewbox_coords_list):
    if not '&' in address_variant:
        return try_nominatim(address_variant, viewbox_coords_list)
    else:
        conn = None
        try:
            conn = connection_pool.getconn()
            parts = address_variant.split('&')
            if len(parts) < 2:
                return None, None, f"Error: Malformed intersection '{address_variant}'"
            street1 = parts[0].split(',')[0].strip()
            street2 = parts[1].split(',')[0].strip()
            return try_postgis_intersection(street1, street2, conn, viewbox_coords_list)
        except Exception as e:
            return None, None, f"Error in PostGIS: {e}"
        finally:
            if conn:
                connection_pool.putconn(conn)



# --- Main geocoding function, goes into futures ---
## assuming pass1 is already done
def process_address(original_address, viewbox_dict):
    city_name = extract_city(original_address)
    viewbox_coords_list = viewbox_dict.get(city_name)
    
    abbr_expanded = expand_abbreviations(original_address)
    dir_expanded  = expand_directions(original_address)
    
    full_expanded = None
    if abbr_expanded:
        full_expanded = expand_directions(abbr_expanded)
    
    queries = [original_address]
    for variant in (abbr_expanded, dir_expanded, full_expanded):
        if variant:
            queries.append(variant)
    
    for query in queries:
        lat, lon, result_msg = _query_geocoders(query, viewbox_coords_list)
        if lat is not None and lon is not None:
            return [original_address, query, lat, lon, result_msg]

    # --- All variants failed, Fuzzy suffix fallback from last query (e.g., C → Circle or Court) ---
    # is this only changing suffix of first found street?
    base_part = query.rsplit(',', 2)[0]
    #base_part = address_pass2.split(',')[:-2]
    match = re.search(r'\b(' + "|".join(FUZZY_SUFFIXES) + r')\b(?=\s*&|\s*,|$)', base_part)
    if match:
        suffix_letter = match.group(1)
        for raw, pattern, replacement in NAME_CLEANUP_MAP:
            if raw == suffix_letter:
                fuzzy_variant = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                if fuzzy_variant != query:
                    lat, lon, result_msg = _query_geocoders(fuzzy_variant, viewbox_coords_list)
                    if lat is not None and lon is not None:
                        return [original_address, fuzzy_variant, lat, lon, result_msg]

    # --- All variants failed found ---
    return [original_address, query, None, None, result_msg]


def main():
    global connection_pool
    viewbox_dict = load_viewboxes(VIEWBOX_FILE)
    
    raw_addresses = []
    with open(INPUT_FILE, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        if "address" not in reader.fieldnames:
            print(f"\u274c Error: 'address' column not found in {INPUT_FILE}")
            return
        raw_addresses = [row["address"] for row in reader]

    cities_in_data = set()
    for addr in raw_addresses:
        city = extract_city(addr)
        if city:
            cities_in_data.add(city)
    
    missing_viewboxes = sorted(c for c in cities_in_data if c not in viewbox_dict)
    if missing_viewboxes:
        print(f"\u274c Error: Missing viewboxes for cities: {', '.join(missing_viewboxes)}")
        return

    connection_pool = pool.ThreadedConnectionPool(1, MAX_WORKERS, **DB_PARAMS)
    
    with open(OUTPUT_FILE_MATCHES, 'w', newline='', encoding='utf-8') as matches_file, \
         open(OUTPUT_FILE_UNMATCHED, 'w', newline='', encoding='utf-8') as unmatched_file:
        
        matches_writer = csv.writer(matches_file)
        unmatched_writer = csv.writer(unmatched_file)
        
        header = ["original_address", "used_variant", "lat", "lon", "result"]
        matches_writer.writerow(header)
        unmatched_writer.writerow(header)
        
        matched_count = 0
        unmatched_count = 0
        total_count = len(raw_addresses)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_address, address, viewbox_dict): address for address in raw_addresses}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Geocoding"):
                # save get original address for potential error
                original_addr = futures[future]
                try:
                    result_row = future.result() # process_address always returns a 5-element list
                    
                    # Check if lat and lon (indices 2 and 3) are not None
                    if result_row[2] is not None and result_row[3] is not None:
                        matches_writer.writerow(result_row)
                        matched_count += 1
                    else:
                        unmatched_writer.writerow(result_row)
                        unmatched_count += 1
                except Exception as e: 
                    used_variant = expand_abbreviations(original_addr)                    
                    error_row = [original_addr, used_variant, None, None, f"Critical Error: {e}"]
                    unmatched_writer.writerow(error_row)
                    unmatched_count +=1 # Count critical errors as unmatched
    
    if connection_pool:
        connection_pool.closeall()
    
    print(f"✅ Finished! Results saved.")
    print(f"   Matched addresses: {matched_count} (saved to {OUTPUT_FILE_MATCHES})")
    print(f"   Unmatched addresses: {unmatched_count} (saved to {OUTPUT_FILE_UNMATCHED})")
    print(f"   Total processed: {total_count}; Match rate: {matched_count / total_count * 100:.2f}%")

if __name__ == "__main__":
    main()
