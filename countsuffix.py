import pandas as pd

df = pd.read_csv('VB_geocoded_pass2_unmatched.csv', dtype=str)

# 1) Pull off the street as a pure string
df['street'] = (
    df['original_address']
      .astype(str)
      .str.rsplit(',', n=2)
      .str[0]
      .str.strip()
)

# 2) Extract the trailing token
df['suffix'] = (
    df['street']
      .str.extract(r'(\w+)$', expand=False)
      .str.upper()
)

# 3) Count
counts = df['suffix'].value_counts().head(20)
print(counts)
