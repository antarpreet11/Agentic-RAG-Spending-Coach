"""
ID generation utilities for transactions
"""

import hashlib
from typing import Union

def txn_id_from_fields(txn_date: str, amount: float, description: str, account_id: str) -> str:
    """
    Generate a unique transaction ID from transaction fields
    
    Args:
        txn_date: Transaction date
        amount: Transaction amount
        description: Transaction description
        account_id: Account identifier
    
    Returns:
        Unique transaction ID (SHA-256 hash)
    """
    # Create a string from the fields
    fields_string = f"{txn_date}|{amount}|{description}|{account_id}"
    
    # Generate SHA-256 hash
    return hashlib.sha256(fields_string.encode('utf-8')).hexdigest()
