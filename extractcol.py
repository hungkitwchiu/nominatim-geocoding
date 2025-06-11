import pandas as pd

input_file = "STP_geocoded_unmatched.csv"
output_file = "STP_address.csv"

df = pd.read_csv(input_file)

# Extract only the 'original_address' column
original_addresses = df[['original_address']]
original_addresses.columns = ["address"]

# Save to a new CSV file
original_addresses.to_csv(output_file, index=False)
