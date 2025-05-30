import pandas as pd

# Load the CSV file
input_file = "VB_geocoded_pass2_unmatched.csv"
df = pd.read_csv(input_file)

# Extract only the 'original_address' column
original_addresses = df[['original_address']]
original_addresses.columns = ["address"]
# Save to a new CSV file
output_file = "VB_address.csv"
original_addresses.to_csv(output_file, index=False)
