from typing import Dict, Any, Tuple
from core.transactions.schema import Txn

def validate_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        Txn.model_validate(row)
        return True, ""
    except Exception as e:
        return False, str(e)
