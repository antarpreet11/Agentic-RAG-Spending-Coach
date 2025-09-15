import os, glob
from dotenv import load_dotenv
from core.common.io import write_csv
from core.transactions.ingest import ingest_files
from logging_config import setup_logging

def run_step1_generalize():
    """Step 1: Process and normalize transaction data from CSV files"""
    load_dotenv()
    log = setup_logging()

    input_glob = os.getenv("INPUT_GLOB", "data/raw/*.csv")
    out_clean = os.getenv("OUT_CLEAN", "data/processed/clean.csv")
    out_quar = os.getenv("OUT_QUAR", "data/processed/quarantine.csv")
    out_filtered = os.getenv("OUT_FILTERED", "data/processed/filtered.csv")
    account_id = os.getenv("ACCOUNT_ID", "default")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    files = sorted(glob.glob(input_glob))
    if not files:
        log.error(f"No files match pattern: {input_glob}")
        return

    log.info(f"Running Step 1 on {len(files)} file(s) with model={model} ...")
    clean_df, bad_df, filtered_df = ingest_files(files, account_id=account_id, model_name=model)

    if not clean_df.empty:
        write_csv(clean_df, out_clean)
        log.info(f"Wrote clean rows: {out_clean} ({len(clean_df)} rows)")
    else:
        log.warning("No valid rows produced.")

    if not bad_df.empty:
        write_csv(bad_df, out_quar)
        log.info(f"Wrote quarantine rows: {out_quar} ({len(bad_df)} rows)")
    else:
        log.info("No quarantine rows.")

    if not filtered_df.empty:
        write_csv(filtered_df, out_filtered)
        log.info(f"Wrote filtered rows: {out_filtered} ({len(filtered_df)} rows)")
    else:
        log.info("No filtered rows.")
