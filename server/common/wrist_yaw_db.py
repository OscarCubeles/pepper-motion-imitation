import os
import csv

# In-memory singleton cache for the CSV lookup
_DATA = None


def _csv_path():
    # CSV is located next to this module
    return os.path.join(os.path.dirname(__file__), "Combinations Hand_final.csv")


def _load_data():
    global _DATA
    if _DATA is not None:
        return
    _DATA = {}
    path = _csv_path()
    try:
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh, delimiter=';')
            for row in reader:
                # skip empty or malformed rows
                if not row or len(row) < 5:
                    continue
                upper = (row[0] or '').strip().upper()
                fore = (row[1] or '').strip().upper()
                palm = (row[2] or '').strip().upper()
                twist_right = (row[3] or '').strip()
                twist_left = (row[4] or '').strip()
                key = (upper, fore, palm)
                _DATA[key] = {'right': twist_right, 'left': twist_left, 'row': row}
    except Exception:
        # If file missing or unreadable, leave empty dict
        _DATA = {}


def query_wrist_twist(upper_arm, forearm, hand_palm, hand='right'):
    """Query the CSV by (UpperArm, Forearm, HandPalm) and return the twist for the given hand.

    Args:
        upper_arm (str): Upper arm label (e.g. 'UP', 'LEFT')
        forearm (str): Forearm label
        hand_palm (str): Hand palm label
        hand (str): 'left' or 'right' (case-insensitive). Defaults to 'right'.

    Returns:
        str or None: The raw CSV cell for the twist (may be 'CENTERED', '360', 'NAN WRIST', numeric string, etc.),
        or None if no matching row exists.
    """
    _load_data()
    key = ((upper_arm or '').strip().upper(), (forearm or '').strip().upper(), (hand_palm or '').strip().upper())
    entry = _DATA.get(key)
    if entry is None:
        #print(f"Wrist yaw DB: No entry found for key {key}")
        return None
    hand_flag = (hand or 'right').strip().lower()
    return entry['left'] if hand_flag.startswith('l') else entry['right']
