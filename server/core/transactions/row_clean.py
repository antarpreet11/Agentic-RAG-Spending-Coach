from typing import Dict, Any, Optional, List, Tuple
from dateparser import parse as dparse
from core.common.text import clean_desc
import math

def parse_date_safe(s: str) -> Optional[str]:
    if not s:
        return None
    dt = dparse(s, settings={"DATE_ORDER": "YMD"})
    return dt.date().isoformat() if dt else None

def join_description(cols: List[str], row: Dict[str, Any]) -> str:
    parts = []
    for c in cols or []:
        parts.append(str(row.get(c, "") or ""))
    return clean_desc(" ".join(parts))

def normalize_amount_and_flow(amount_str: str, flow_val: Optional[str], flow_debit_values=None, flow_credit_values=None) -> Tuple[float,str]:
    flow_debit_values = set([v.lower() for v in (flow_debit_values or [])])
    flow_credit_values = set([v.lower() for v in (flow_credit_values or [])])

    # Coerce numeric
    s = (amount_str or "").replace(",", "")
    try:
        amt = float(s)
    except Exception:
        # If not parseable treat as NaN to be validated out later
        return (math.nan, "debit")

    # Sign-based default
    flow = "debit" if amt < 0 else "credit"
    amt = abs(amt)

    # Override with explicit flow column if known
    if flow_val:
        v = str(flow_val).strip().lower()
        if v in flow_debit_values:
            flow = "debit"
        elif v in flow_credit_values:
            flow = "credit"

    return (amt, flow)
