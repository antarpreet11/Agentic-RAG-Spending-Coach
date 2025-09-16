"""
Text processing utilities
"""

import re
from typing import Optional

def clean_desc(description: str) -> str:
    """
    Clean transaction description text
    
    Args:
        description: Raw transaction description
    
    Returns:
        Cleaned description
    """
    if not description:
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', description.strip())
    
    # Remove common prefixes/suffixes that don't add value
    cleaned = re.sub(r'^[A-Z]{2,4}\s*\*', '', cleaned)  # Remove merchant codes like "SQ *"
    cleaned = re.sub(r'\s+\|\s+.*$', '', cleaned)  # Remove everything after "|"
    
    return cleaned.strip()
