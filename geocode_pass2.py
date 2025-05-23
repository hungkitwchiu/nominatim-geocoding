import csv
import requests
from psycopg2 import pool
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configuration
#INPUT_FILE = "geocoded_unmatched.csv"
INPUT_FILE = "CAM_address.csv"
VIEWBOX_FILE = "city_viewboxes.csv"
OUTPUT_FILE_MATCHES = "CAM_geocoded_pass2_matches.csv"
OUTPUT_FILE_UNMATCHED = "CAM_geocoded_pass2_unmatched.csv"
MAX_WORKERS = 10
API_URL = "http://localhost/nominatim/search"

DB_PARAMS = {
    'dbname': 'osm_raw',
    'user': 'nominatim',
    'host': 'localhost',
    'password': 'nominatim'
}

BUFFER_DISTANCE = 500
connection_pool = None


# --- ABBREVIATION AND DIRECTION DEFINITIONS ---

def detect_directions(address_text):
    for pattern in DIRECTION_MAP.keys():
        if re.search(pattern, address_text, flags=re.IGNORECASE):
            return True
    return False


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
    # viewbox_coords_list should always be there, as code won't get here if not
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

def try_postgis_intersection(street1, street2, db_conn, viewbox_coords_list):
    # viewbox_coords_list should always be there, as code won't get here if not
    if viewbox_coords_list is None: return None, None, "Skipped: No viewbox for PostGIS"
    left, top, right, bottom = viewbox_coords_list
    pg_min_lon, pg_min_lat, pg_max_lon, pg_max_lat = left, bottom, right, top

    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS(SELECT 1 FROM planet_osm_line WHERE unaccent(name) ILIKE unaccent(%s) AND ST_Intersects(way, ST_MakeEnvelope(%s,%s,%s,%s,4326))) AS e1,
                   EXISTS(SELECT 1 FROM planet_osm_line WHERE unaccent(name) ILIKE unaccent(%s) AND ST_Intersects(way, ST_MakeEnvelope(%s,%s,%s,%s,4326))) AS e2;
        """, (f"%{street1}%", pg_min_lon, pg_min_lat, pg_max_lon, pg_max_lat, f"%{street2}%", pg_min_lon, pg_min_lat, pg_max_lon, pg_max_lat))
        exists1, exists2 = cur.fetchone()
        if not (exists1 and exists2): return None, None, f"'{street1}' or '{street2}' not in viewbox"

        cur.execute("""
            SELECT ST_Y(pt) AS lat, ST_X(pt) AS lon
            FROM (
                SELECT (ST_Dump(ST_Intersection(w1.way, w2.way))).geom AS pt
                FROM planet_osm_line w1, planet_osm_line w2
                WHERE unaccent(w1.name) ILIKE unaccent(%s)
                AND unaccent(w2.name) ILIKE unaccent(%s)
                AND ST_Intersects(w1.way, w2.way)
                AND ST_Intersects(w2.way, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
            ) AS dumped
            WHERE GeometryType(pt) = 'POINT'
            LIMIT 1;
        """, (
            f"%{street1}%", f"%{street2}%",
            pg_min_lon, pg_min_lat, pg_max_lon, pg_max_lat
        ))
        row = cur.fetchone()
        if row and row[0] is not None: return row[0], row[1], f"PostGIS: Intersection {street1} & {street2}"

        cur.execute("""
            SELECT ST_Y(ST_Centroid(ST_Union(w1.way))), ST_X(ST_Centroid(ST_Union(w1.way)))
            FROM planet_osm_line w1 JOIN planet_osm_line w2 ON ST_DWithin(w1.way, w2.way, %s)
            WHERE unaccent(w1.name) ILIKE unaccent(%s) AND unaccent(w2.name) ILIKE unaccent(%s)
            AND ST_Intersects(w1.way, ST_MakeEnvelope(%s,%s,%s,%s,4326)) AND ST_Intersects(w2.way, ST_MakeEnvelope(%s,%s,%s,%s,4326))
            GROUP BY w1.osm_id, w2.osm_id LIMIT 1;
        """, (BUFFER_DISTANCE, f"%{street1}%", f"%{street2}%", pg_min_lon, pg_min_lat, pg_max_lon, pg_max_lat, pg_min_lon, pg_min_lat, pg_max_lon, pg_max_lat))
        row = cur.fetchone()
        if row and row[0] is not None: return row[0], row[1], f"PostGIS: Approx centroid {street1} & {street2}"
    return None, None, "No PostGIS match"
# --- END OF UTILITY AND QUERY FUNCTIONS ---

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
def process_address(original_address, viewbox_dict):
    address_pass1 = expand_abbreviations(original_address)
    city_name = extract_city(address_pass1)
    viewbox_coords_list = viewbox_dict.get(city_name)

    lat, lon, result_msg = _query_geocoders(address_pass1, viewbox_coords_list)

    if lat is not None and lon is not None:
        return [original_address, address_pass1, lat, lon, result_msg]

    # --- Direction expansion fallback ---
    if detect_directions(address_pass1):
        address_pass2 = expand_directions(address_pass1)
        lat, lon, result_msg = _query_geocoders(address_pass2, viewbox_coords_list)
        if lat is not None and lon is not None:
            return [original_address, address_pass2, lat, lon, result_msg]

    # --- Fuzzy suffix fallback (e.g., C → Circle or Court) ---
    # is this only changing suffix of first found street?
    base_part = address_pass1.split(',')[:-2]
    match = re.search(r'\b(' + "|".join(FUZZY_SUFFIXES) + r')\b(?=\s*&|\s*,|$)', test_address)
    if match:
        suffix_letter = match.group(1)
        for raw, pattern, replacement in NAME_CLEANUP_MAP:
            if raw == suffix_letter:
                fuzzy_variant = re.sub(pattern, replacement, address_pass1, flags=re.IGNORECASE)
                if fuzzy_variant != address_pass1:
                    lat, lon, result_msg = _query_geocoders(fuzzy_variant, viewbox_coords_list)
                    if lat is not None and lon is not None:
                        return [original_address, fuzzy_variant, lat, lon, result_msg]

    # --- No match found ---
    return [original_address, address_pass1, None, None, result_msg]


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
