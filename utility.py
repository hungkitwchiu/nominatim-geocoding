import csv

FUZZY_SUFFIXES = ['C', 'H', 'P', 'L']
NAME_CLEANUP_MAP = load_cleanup_rules("name_cleanup_rules.csv")

def load_cleanup_rules(csv_path):
    cleanup_list = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_match = row['match'].strip()
            replacement = row['replace']
            if not raw_match:
                continue

            if raw_match in ['St', 'Mt']:
                pattern = rf'\b{raw_match}\s+(?=[A-Z])'
            else:
                pattern = fr'\b{raw_match}\b'

            # Append tuple: (raw_match, regex_pattern, replacement)
            cleanup_list.append((raw_match, pattern, replacement))
    return cleanup_list

def expand_abbreviations(address):
    parts = [p.strip() for p in address.split(',')]
    base = ','.join(parts[:-2]) if len(parts) >= 3 else address
    suffix = ', ' + parts[-2] + ', ' + parts[-1] if len(parts) >= 3 else ""

    for raw, pattern, replacement in NAME_CLEANUP_MAP:
        if raw in FUZZY_SUFFIXES:  # skip fuzzy suffixes
            continue
        base = re.sub(pattern, replacement, base, flags=re.IGNORECASE)

    base = re.sub(r'\bSt(?=(\s&|&|,|$))', 'Street', base, flags=re.IGNORECASE)
    base = re.sub(r'^\s*&\s*|\s*&\s*$', '', base)
    return re.sub(r'\s+', ' ', base).strip() + suffix

def expand_directions(address):
    for pattern, replacement in DIRECTION_MAP.items():
        expanded_address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
    expanded_address = re.sub(r'\s+', ' ', expanded_address).strip()
    return expanded_address
