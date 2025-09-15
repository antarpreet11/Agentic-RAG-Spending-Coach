from typing import Dict, List, Optional

# Deterministic header maps for known banks (as a starting point)
# Keys are lowercase header names.
SCOTIA_MAP = {
    "date": "txn_date",
    "sub-description": "description_extra",
    "description": "description_main",
    "status": "status",
    "type of transaction": "flow_column",
    "amount": "amount",
}

NEO_MAP = {
    "transaction date": "txn_date",
    "posted date": "posted_date",
    "status": "status",
    "description": "description_main",
    "amount": "amount",
}

KNOWN_BANKS = {
    "scotiabank": SCOTIA_MAP,
    "neo": NEO_MAP,
}

def detect_known_bank(headers_lower: List[str]) -> Optional[str]:
    hs = set(headers_lower)
    if {"date","description","sub-description","status","type of transaction","amount"}.issubset(hs):
        return "scotiabank"
    if {"transaction date","posted date","status","description","amount"}.issubset(hs):
        return "neo"
    return None

def rules_mapping_for_bank(bank: str) -> Dict[str, str]:
    return KNOWN_BANKS.get(bank, {})
