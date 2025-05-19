import csv
import requests
from psycopg2 import pool
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configuration
INPUT_FILE = "geocoded_unmatched.short.csv"
OUTPUT_MATCHED = "geocoded_pass2_matches.csv"
OUTPUT_UNMATCHED = "geocoded_pass2_unmatched.csv"
VIEWBOX_FILE = "city_viewboxes.csv"

NUM_WORKERS = 5
API_URL = "http://localhost/nominatim/search"

DB_PARAMS = {
    'dbname': 'osm_raw',
    'user': 'nominatim',
    'host': 'localhost',
    'password': 'nominatim' # use your own database password here
}

BUFFER_DISTANCE = 500  # meters
connection_pool = None

ABBREVIATION_MAP = {
    r'\bST\b': 'Street',
    r'\bDR\b': 'Drive',
    r'\bD\b': 'Drive',
    r'\bAV\b': 'Avenue',
    r'\bAVE\b': 'Avenue',
    r'\bRD\b': 'Road',
    r'\bBL\b': 'Boulevard',
    r'\bCT\b': 'Court',
    r'\bCL\b': 'Circle',
    r'\bCIR\b': 'Circle',
    r'\bC\b': 'Circle',
    r'\bCTR\b': 'Center',
    r'\bLN\b': 'Lane',
    r'\bWY\b': 'Way',
    r'\bPL\b': 'Place',
    r'\bEX\b': 'Expressway',
    r'\bTER\b': 'Terrace',
    r'\bSQ\b': 'Square',
    r'\bPKWY\b': 'Parkway',
    r'\bHWY\b': 'Highway'
}

DIRECTION_MAP = {
    r'(^|[\s,&])N(?=\s)': r'\1North',
    r'(^|[\s,&])E(?=\s)': r'\1East',
    r'(^|[\s,&])S(?=\s)': r'\1South',
    r'(^|[\s,&])W(?=\s)': r'\1West'
}

def expand_abbreviations(address):
    for pattern, replacement in DIRECTION_MAP.items():
        address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
    for abbrev, full in ABBREVIATION_MAP.items():
        address = re.sub(abbrev, full, address, flags=re.IGNORECASE)
    return address

def load_viewboxes(file_path):
    viewbox_dict = {}
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            viewbox = row['viewbox']
            if parse_viewbox(viewbox) is None:
                raise ValueError(f"Malformed viewbox for city '{row['city']}'")
            viewbox_dict[row['city'].lower()] = viewbox
    return viewbox_dict

def extract_city(address):
    parts = address.split(',')
    return parts[1].strip().lower() if len(parts) >= 2 else None

def parse_viewbox(viewbox_str):
    try:
        parts = list(map(float, viewbox_str.split(',')))
        return parts if len(parts) == 4 else None
    except Exception:
        return None

def try_postgis_intersection(street1, street2, db_conn, viewbox_coords):
    minLon, minLat, maxLon, maxLat = viewbox_coords
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT
              EXISTS(
                SELECT 1 FROM planet_osm_line
                WHERE unaccent(name) ILIKE unaccent(%s)
                  AND ST_Intersects(way, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
              ) AS exists1,
              EXISTS(
                SELECT 1 FROM planet_osm_line
                WHERE unaccent(name) ILIKE unaccent(%s)
                  AND ST_Intersects(way, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
              ) AS exists2;
        """, (
            f"%{street1}%", minLon, minLat, maxLon, maxLat,
            f"%{street2}%", minLon, minLat, maxLon, maxLat
        ))
        exists1, exists2 = cur.fetchone()
        if not (exists1 and exists2):
            return None, None, f"Skipped: '{street1}' or '{street2}' not found (fuzzy match failed)"

        cur.execute("""
            SELECT ST_Y(geom), ST_X(geom)
            FROM (
                SELECT ST_Intersection(w1.way, w2.way) AS geom
                FROM planet_osm_line w1, planet_osm_line w2
                WHERE unaccent(w1.name) ILIKE unaccent(%s)
                  AND unaccent(w2.name) ILIKE unaccent(%s)
                  AND ST_Intersects(w1.way, w2.way)
                  AND ST_Intersects(w1.way, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
                  AND ST_Intersects(w2.way, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
            ) AS sub
            WHERE ST_IsValid(geom)
            LIMIT 1;
        """, (
            f"%{street1}%", f"%{street2}%",
            minLon, minLat, maxLon, maxLat,
            minLon, minLat, maxLon, maxLat
        ))
        row = cur.fetchone()
        if row and row[0] is not None and row[1] is not None:
            return row[0], row[1], f"Intersection of {street1} & {street2}"

        cur.execute("""
            SELECT ST_Y(ST_Centroid(ST_Union(w1.way))),
                   ST_X(ST_Centroid(ST_Union(w1.way)))
            FROM planet_osm_line w1, planet_osm_line w2
            WHERE unaccent(w1.name) ILIKE unaccent(%s)
              AND unaccent(w2.name) ILIKE unaccent(%s)
              AND ST_DWithin(w1.way, w2.way, %s)
              AND ST_Intersects(w1.way, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
              AND ST_Intersects(w2.way, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
        """, (
            f"%{street1}%", f"%{street2}%", BUFFER_DISTANCE,
            minLon, minLat, maxLon, maxLat,
            minLon, minLat, maxLon, maxLat
        ))
        row = cur.fetchone()
        if row and row[0] is not None and row[1] is not None:
            return row[0], row[1], f"Approximate centroid of near-match for {street1} & {street2}"
    return None, None, "No match"

def try_nominatim_query(cleaned_address, viewbox_coords):
    try:
        minLon, minLat, maxLon, maxLat = viewbox_coords
        viewbox_str = f"{minLon},{maxLat},{maxLon},{minLat}"
        response = requests.get(API_URL, params={
            "q": cleaned_address,
            "format": "json",
            "limit": 1,
            "viewbox": viewbox_str,
            "bounded": 1
        })
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), "Single-point address match (bounded)"
        else:
            return None, None, "No match from Nominatim"
    except Exception as e:
        return None, None, f"Nominatim error: {e}"

def process_address(original_address, viewbox_dict):
    address = expand_abbreviations(original_address)
    city = extract_city(address)
    viewbox_str = viewbox_dict.get(city)
    viewbox_coords = parse_viewbox(viewbox_str)
    if viewbox_coords is None:
        return [original_address, address, None, None, f"Error: malformed viewbox for city '{city}'"]

    if '&' not in address:
        lat, lon, result = try_nominatim_query(address, viewbox_coords)
        return [original_address, address, lat, lon, result]

    conn = None
    try:
        conn = connection_pool.getconn()
        parts = address.split('&')
        street1 = parts[0].split(',')[0].strip()
        street2 = parts[1].split(',')[0].strip()
        lat, lon, result = try_postgis_intersection(street1, street2, conn, viewbox_coords)
        return [original_address, address, lat, lon, result]
    except Exception as e:
        return [original_address, address, None, None, f"Error: {e}"]
    finally:
        if conn:
            connection_pool.putconn(conn)

def main():
    global connection_pool
    viewbox_dict = load_viewboxes(VIEWBOX_FILE)

    with open(INPUT_FILE, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        raw_addresses = [row["address"] for row in reader]

    cities = {extract_city(expand_abbreviations(addr)) for addr in raw_addresses}
    missing = sorted(c for c in cities if c not in viewbox_dict)
    if missing:
        print(f"\u274c Error: Missing viewboxes for cities: {', '.join(missing)}")
        return

    connection_pool = pool.ThreadedConnectionPool(1, NUM_WORKERS, **DB_PARAMS)

    with open(OUTPUT_MATCHED, 'w', newline='', encoding='utf-8') as matched_file, \
         open(OUTPUT_UNMATCHED, 'w', newline='', encoding='utf-8') as unmatched_file:

        match_writer = csv.writer(matched_file)
        unmatch_writer = csv.writer(unmatched_file)

        headers = ["original_address", "used_variant", "lat", "lon", "result"]
        match_writer.writerow(headers)
        unmatch_writer.writerow(headers)

        matched_count = 0
        total_count = len(raw_addresses)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_address, address, viewbox_dict): address
                for address in raw_addresses
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Geocoding"):
                try:
                    result_row = future.result()
                    if result_row[2] is not None and result_row[3] is not None:
                        match_writer.writerow(result_row)
                        matched_count += 1
                    else:
                        unmatch_writer.writerow(result_row)
                except Exception as e:
                    addr = futures[future]
                    unmatch_writer.writerow([addr, expand_abbreviations(addr), None, None, f"Error: {e}"])

    connection_pool.closeall()
    print(
        f"✅ Pass 2 Complete. {matched_count} out of {total_count} addresses matched "
        f"({matched_count / total_count * 100:.2f}%). {total_count - matched_count} unmatched."
    )

if __name__ == "__main__":
    main()
