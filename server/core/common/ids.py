import hashlib

def txn_id_from_fields(date_str: str, amount: float, description: str, account_id: str) -> str:
    key = f"{date_str}|{amount:.2f}|{description.strip()}|{account_id.strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()
