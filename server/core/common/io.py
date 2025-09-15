import pandas as pd
from pathlib import Path

def read_csv_loose(path: str) -> pd.DataFrame:
    # A minimal loose CSV reader (utf-8) with dtype guessing
    # Handle malformed CSV files with inconsistent field counts
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.ParserError as e:
        # If parsing fails due to inconsistent field counts, try with error handling
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8", 
                             on_bad_lines='skip', engine='python')
        except Exception:
            # Last resort: try with different parameters
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8",
                             sep=',', quotechar='"', on_bad_lines='skip', engine='python')

def write_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
