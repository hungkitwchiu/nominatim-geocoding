import csv
import re

DIRECTION_MAP = {
    r'(^|[\s,&])N(?=(\s|,))': r'\1North',
    r'(^|[\s,&])E(?=(\s|,))': r'\1East',
    r'(^|[\s,&])S(?=(\s|,))': r'\1South',
    r'(^|[\s,&])W(?=(\s|,))': r'\1West',
    r'(^|[\s,&])NE(?=(\s|,))': r'\1Northeast',
    r'(^|[\s,&])SE(?=(\s|,))': r'\1Southeast',
    r'(^|[\s,&])NW(?=(\s|,))': r'\1Northwest',
    r'(^|[\s,&])SW(?=(\s|,))': r'\1Southwest'
}
FUZZY_SUFFIXES = ['C', 'P', 'L', 'S']

def load_cleanup_rules(csv_path):
    cleanup_list = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_match = row['match'].strip()
            replacement = row['replace']
            note = row['notes']
            if not raw_match:
                continue
            # St followed by 1 letter could actually be "street {direction}"
            if raw_match in ['St', 'Mt']:
                pattern = fr'\b{raw_match}\b(?=\s+[A-Za-z]{{2,}})'
            elif "no-boundary" in note:
                pattern = fr'{raw_match}'
            else:
                pattern = fr'\b{raw_match}\b'

            # Append tuple: (raw_match, regex_pattern, replacement)
            cleanup_list.append((raw_match, pattern, replacement, note))
    return cleanup_list
    
def load_viewboxes(file_path):
    VIEWBOX_DICT = {}
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            viewbox_str = row['viewbox']
            parts = list(map(float, viewbox_str.split(',')))
            parsed_coords = parts if len(parts) == 4 else None
            if parsed_coords is None:
                raise ValueError(f"Malformed viewbox for city '{row['city']}': '{viewbox_str}'")
            VIEWBOX_DICT[row['city'].lower()] = parsed_coords
    return VIEWBOX_DICT


NAME_CLEANUP_MAP = load_cleanup_rules("name_cleanup_rules.csv")
VIEWBOX_DICT = load_viewboxes("city_viewboxes.csv")

def expand_abbreviations(address):
    parts = [p.strip() for p in address.split(',')]
    base  = ','.join(parts[:-2]) if len(parts) >= 3 else address
    other = ', ' + parts[-2] + ', ' + parts[-1] if len(parts) >= 3 else ''

    for raw, pattern, replacement, note in NAME_CLEANUP_MAP:
        if raw in FUZZY_SUFFIXES:
            continue
        base = re.sub(pattern, replacement, base, flags=re.IGNORECASE)

    base = re.sub(r'\bSt(?=(\s|&|,|/|$))', 'Street', base, flags=re.IGNORECASE)
    base = re.sub(r'^\s*&\s*|\s*&\s*$', '', base)
    expanded = re.sub(r'\s+', ' ', base).strip() + other

    # only return a string if it actually changed
    if expanded.lower() == address.lower():
        return None
    return expanded

def remove_suffix(address):
    # address should not contain city coming into here, should be XX Street, State
    parts  = [p.strip() for p in address.split(',')]
    base   = ','.join(parts[:-1]) if len(parts) >= 2 else address
    suffix = ', ' + parts[-1] if len(parts) >= 2 else ''
    
    for raw, pattern, replacement, note in NAME_CLEANUP_MAP:
        #if raw in FUZZY_SUFFIXES:
        #    continue
        # only remove suffixes type of expansion, so names/typos, etc untouched
        if 'suffix' not in note.lower():
            continue
        elif 'fuzzy suffix' == note.lower():
            base = re.sub(fr'\b{raw}(?=(\s&|&|,|/|$))', "", base, flags=re.IGNORECASE)
        base = re.sub(fr'\b{replacement}(?=(\s&|&|,|/|$))', "", base, flags=re.IGNORECASE)
    
    # removed = re.sub(r'\bStreet(?=(\s|&|,|$))', '', base, flags=re.IGNORECASE)    
    removed = re.sub(r'\s+', ' ', base).strip() + suffix
    if removed.lower() == address.lower():
        return None
    return removed

def expand_directions(address):
    expanded = address
    for pattern, replacement in DIRECTION_MAP.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    expanded = re.sub(r'\s+', ' ', expanded).strip()

    if expanded.lower() == address.lower():
        return None
    return expanded
