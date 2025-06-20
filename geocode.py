import csv
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from geofunctions import load_cleanup_rules
from geofunctions import expand_abbreviations
from geofunctions import expand_directions
from geofunctions import NAME_CLEANUP_MAP

INPUT_FILE = "address.csv"
OUTPUT_MATCHES = "geocoded_matches.csv"
OUTPUT_UNMATCHED = "geocoded_unmatched.csv"

API_URL = "http://localhost/nominatim/search"
MAX_WORKERS = 10

session = requests.Session()

def geocode_address(original_address):
    abbr_expanded = expand_abbreviations(original_address)
    dir_expanded  = expand_directions(original_address)

    full_expanded = None
    if abbr_expanded:
        full_expanded = expand_directions(abbr_expanded)

    # collect only the non‐None queries, original first
    queries = [original_address]
    for variant in (abbr_expanded, dir_expanded, full_expanded):
        if variant:
            queries.append(variant)

    for query in queries:
        params = {
            'q': query,
            'format': 'json',
            'addressdetails': 1,
            'limit': 1
        }
        try:
            resp = session.get(API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return original_address, query, None, None, f"Error: {e}"

        if data:
            lat  = data[0]['lat']
            lon  = data[0]['lon']
            name = data[0]['display_name']
            return original_address, query, lat, lon, name

    return original_address, None, None, None, "No match"
        

def main():
    with open(INPUT_FILE, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        addresses = [row["address"] for row in reader]

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(geocode_address, addr): addr for addr in addresses}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 Pass 1 Geocoding"):
            results.append(future.result())
            
    matched_count = 0
    total_count = len(results)

    # Write outputs
    with open(OUTPUT_MATCHES, 'w', newline='', encoding='utf-8') as mfile, \
         open(OUTPUT_UNMATCHED, 'w', newline='', encoding='utf-8') as umfile:

        match_writer = csv.writer(mfile)
        unmatched_writer = csv.writer(umfile)

        headers = ["original_address", "used_variant", "lat", "lon", "result"]
        match_writer.writerow(headers)
        unmatched_writer.writerow(["address"])

        for row in results:
            original, used_variant, lat, lon, result = row
            if result == "No match" or result.startswith("Error"):
                unmatched_writer.writerow([original])
            else:
                match_writer.writerow(row)
                matched_count += 1
    print(
        f"✅ Pass 1 Complete. {matched_count} out of {total_count} addresses matched "
        f"({matched_count / total_count * 100:.2f}%) {total_count - matched_count} to go."
    )
if __name__ == "__main__":
    main()
