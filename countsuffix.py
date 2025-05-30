import pandas as pd

df = pd.read_csv('STP_geocoded_pass2_unmatched.csv', dtype=str)

# 1) Pull off the street as a pure string
df['street'] = (
    df['original_address']
      .astype(str)
      .str.rsplit(',', n=2) # strip off last two comma-separated pieces
      .str[0]
      .str.strip()
)

# 2) Extract the trailing token
df['suffix'] = (
    df['street']
      .str.extract(r'(\w+)$', expand=False)
      .str.upper()
)

# 3) Load the list of already‐tried matches and uppercase them
rules_df = pd.read_csv('name_cleanup_rules.csv', dtype=str)
tried = set(rules_df['match'].str.upper())

# 4) Filter out suffixes you’ve already covered
remaining = df[~df['suffix'].isin(tried)]

# 5) Count and print the top 20
counts = remaining['suffix'].value_counts().head(20)
print(counts)
