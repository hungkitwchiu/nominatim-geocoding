import csv
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FILE = "/srv/nominatim-project/address.csv"
OUTPUT_MATCHES = "/srv/nominatim-project/geocoded_matches.csv"
OUTPUT_UNMATCHED = "/srv/nominatim-project/geocoded_unmatched.csv"

API_URL = "http://localhost/nominatim/search"
MAX_WORKERS = 10

def geocode_address(original_address):
    params = {
        'q': original_address,
        'format': 'json',
        'addressdetails': 1,
        'limit': 1
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            lat = data[0]["lat"]
            lon = data[0]["lon"]
            result = data[0]["display_name"]
            return original_address, original_address, lat, lon, result
        else:
            return original_address, original_address, None, None, "No match"
    except Exception as e:
        return original_address, original_address, None, None, f"Error: {e}"

def main():
    with open(INPUT_FILE, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        addresses = [row["address"] for row in reader]

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(geocode_address, addr): addr for addr in addresses}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 Pass 1 Geocoding"):
            results.append(future.result())

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

    print("✅ Geocoding complete.")

if __name__ == "__main__":
    main()
