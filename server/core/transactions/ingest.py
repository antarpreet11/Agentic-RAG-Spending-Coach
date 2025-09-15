from typing import List, Dict, Any, Tuple
import pandas as pd
from core.common.io import read_csv_loose, write_csv
from core.common.ids import txn_id_from_fields
from core.transactions.row_clean import parse_date_safe, join_description, normalize_amount_and_flow
from core.transactions.validate import validate_row
from core.transactions.headers_rules import detect_known_bank, rules_mapping_for_bank
from core.transactions.header_map_llm import map_headers_with_llm, HeaderMap
from loguru import logger

def _headers_lower(df: pd.DataFrame) -> List[str]:
    return [c.strip().lower() for c in df.columns]

def _filter_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out unwanted transactions like payments, reversals, etc.
    Returns: (filtered_df, removed_df)
    """
    original_count = len(df)
    logger.info(f"Filtering transactions: {original_count} rows")
    
    removed_transactions = []
    
    # 1. Remove credit card payments
    payment_patterns = [
        'payment received',
        'payment - thank you',
        'thank you for your payment',
        'payment received - thank you',
        'payment',
        'pay',
        'transfer',
        'deposit'
    ]
    
    # Remove payments (but keep payment-related fees/charges)
    payment_mask = (
        (df['flow'] == 'credit') & 
        df['description'].str.contains('|'.join(payment_patterns), case=False, na=False) &
        (~df['description'].str.contains('fee|rate|installment|charge|interest', case=False, na=False))
    )
    
    payments_removed = df[payment_mask].copy()
    payments_removed['filter_reason'] = 'credit_card_payment'
    removed_transactions.append(payments_removed)
    df_filtered = df[~payment_mask]
    
    # 2. Remove small credit adjustments (likely reversals)
    small_credits = df_filtered[
        (df_filtered['flow'] == 'credit') & 
        (df_filtered['amount'] < 1.0) &
        df_filtered['description'].str.contains('adjustment|credit', case=False, na=False)
    ].copy()
    small_credits['filter_reason'] = 'small_credit_adjustment'
    removed_transactions.append(small_credits)
    df_filtered = df_filtered[~df_filtered.index.isin(small_credits.index)]
    
    # 3. Remove transaction reversals (same amount, same description, opposite flow)
    df_sorted = df_filtered.sort_values(['description', 'amount', 'txn_date'])
    reversals_to_remove = set()
    
    for i in range(len(df_sorted)-1):
        curr = df_sorted.iloc[i]
        next_row = df_sorted.iloc[i+1]
        
        # Check if this looks like a reversal
        if (curr['description'] == next_row['description'] and 
            curr['amount'] == next_row['amount'] and
            curr['flow'] != next_row['flow'] and
            abs((pd.to_datetime(next_row['txn_date']) - pd.to_datetime(curr['txn_date'])).days) <= 30):
            
            # Remove both the original and reversal
            reversals_to_remove.add(curr.name)
            reversals_to_remove.add(next_row.name)
    
    reversals_removed = df_filtered[df_filtered.index.isin(reversals_to_remove)].copy()
    reversals_removed['filter_reason'] = 'transaction_reversal'
    removed_transactions.append(reversals_removed)
    df_filtered = df_filtered[~df_filtered.index.isin(reversals_to_remove)]
    
    # 4. Remove very small transactions (likely fees or noise)
    small_txns = df_filtered[df_filtered['amount'] < 0.50].copy()
    small_txns['filter_reason'] = 'very_small_amount'
    removed_transactions.append(small_txns)
    df_filtered = df_filtered[df_filtered['amount'] >= 0.50]
    
    # Combine all removed transactions
    removed_df = pd.concat(removed_transactions, ignore_index=True) if removed_transactions else pd.DataFrame()
    
    final_count = len(df_filtered)
    removed = original_count - final_count
    logger.info(f"Filtered: {original_count} → {final_count} transactions ({removed} removed)")
    
    return df_filtered, removed_df

def _extract_card_name_from_filename(filename: str) -> str:
    """Extract card name from filename and format it properly.
    
    Examples:
    - amazon_mastercard_1.csv -> Amazon Mastercard 1
    - neo_mastercard_2.csv -> Neo Mastercard 2
    - scotiabank_scene_visa.csv -> Scotiabank Scene Visa
    """
    if not filename:
        return "Unknown Card"
    
    # Get just the filename without path and extension
    import os
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Replace underscores with spaces and title case
    card_name = name_without_ext.replace('_', ' ').title()
    
    return card_name

def _apply_rules_or_llm(df: pd.DataFrame, model_name: str, filename: str = "") -> Tuple[str, HeaderMap]:
    headers = _headers_lower(df)
    
    bank = detect_known_bank(headers)
    if bank:
        logger.info(f"Detected bank: {bank}")
        rule_map = rules_mapping_for_bank(bank)
        # Convert rules into HeaderMap-like structure
        # For simplicity, pick description columns
        if bank == "scotiabank":
            hm = HeaderMap(
                txn_date="Date",
                posted_date=None,
                amount="Amount",
                description_columns=["Description","Sub-description"],
                status="Status",
                flow_column="Type of Transaction",
                flow_debit_values=["Debit"],
                flow_credit_values=["Credit"],
                currency_literal="CAD",
            )
            return bank, hm
        if bank == "neo":
            hm = HeaderMap(
                txn_date="Transaction Date",
                posted_date="Posted Date",
                amount="Amount",
                description_columns=["Description"],
                status="Status",
                flow_column=None,
                flow_debit_values=[],
                flow_credit_values=[],
                currency_literal="CAD",
            )
            return bank, hm

    # Unknown → LLM mapping
    logger.info(f"Unknown bank format - using LLM mapping")
    
    # Extract card name directly from filename
    card_name = _extract_card_name_from_filename(filename)
    logger.info(f"Card name: {card_name}")
    
    result = map_headers_with_llm(list(df.columns), model_name=model_name)
    return card_name, result

def ingest_files(paths: List[str], account_id: str = "default", model_name: str = "gemini-2.0-flash-lite") -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean_rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []

    for path in paths:
        logger.info(f"Processing: {path}")
        df = read_csv_loose(path)
        logger.info(f"  {len(df)} rows, {len(df.columns)} columns")
        
        bank, header_map = _apply_rules_or_llm(df, model_name=model_name, filename=path)

        for idx, row in df.iterrows():
            r = row.to_dict()

            # Skip obvious non-data rows (e.g., Scotia banner line in first row)
            # Try to parse txn date first; if it's missing/invalid, skip to quarantine.
            txn_date_raw = r.get(header_map.txn_date, "")
            txn_date = parse_date_safe(txn_date_raw)
            if not txn_date:
                # push to quarantine with reason and continue
                bad = {"reason": f"Unparseable txn_date from '{txn_date_raw}'", **r}
                bad_rows.append(bad)
                continue

            posted_date = None
            if header_map.posted_date:
                posted_date = parse_date_safe(r.get(header_map.posted_date, ""))

            desc = join_description(header_map.description_columns, r)

            amount_str = str(r.get(header_map.amount, "")).strip()
            flow_col_val = r.get(header_map.flow_column) if header_map.flow_column else None
            amount, flow = normalize_amount_and_flow(amount_str, flow_col_val, header_map.flow_debit_values, header_map.flow_credit_values)

            currency = header_map.currency_literal or "CAD"

            candidate = {
                "txn_id": txn_id_from_fields(txn_date, amount, desc, account_id),
                "account_id": account_id,
                "source_bank": bank,
                "txn_date": txn_date,
                "posted_date": posted_date,
                "description": desc,
                "status": str(r.get(header_map.status, "") or None) if header_map.status else None,
                "flow": flow,
                "amount": amount,
                "currency": currency,
            }

            ok, reason = validate_row(candidate)
            if ok:
                clean_rows.append(candidate)
            else:
                bad = {"reason": reason, **r}
                bad_rows.append(bad)

    clean_df = pd.DataFrame(clean_rows)
    bad_df = pd.DataFrame(bad_rows)
    if not clean_df.empty:
        clean_df = clean_df.sort_values(["txn_date","amount"]).reset_index(drop=True)
    
    logger.info(f"Processed: {len(clean_df)} clean, {len(bad_df)} quarantine")
    
    # Apply additional filtering to clean data
    filtered_df = pd.DataFrame()
    if not clean_df.empty:
        clean_df, filtered_df = _filter_transactions(clean_df)
    
    return clean_df, bad_df, filtered_df
