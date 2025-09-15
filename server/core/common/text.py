import re

def clean_desc(desc: str) -> str:
    desc = (desc or "").strip()
    desc = re.sub(r"\s{2,}", " ", desc)
    return desc
