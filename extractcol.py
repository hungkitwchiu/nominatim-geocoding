import pandas as pd

# Load the CSV file
input_file = "STP_geocoded_pass2_unmatched.csv"
df = pd.read_csv(input_file)

# Extract only the 'original_address' column
original_addresses = df[['original_address']]
original_addresses.columns = ["address"]
# Save to a new CSV file
output_file = "STP_working.csv"
original_addresses.to_csv(output_file, index=False)
