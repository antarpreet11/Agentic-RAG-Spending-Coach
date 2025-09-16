#!/usr/bin/env python3
"""
Agentic RAG Spending Coach - Main API Module
Provides functions for transaction processing and categorization
"""

import os
import glob
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from core.common.io import read_csv_loose, write_csv
from core.transactions.ingest import ingest_files
from core.categorization.classifier import EnsembleClassifier
from logging_config import setup_logging
from loguru import logger

def generalize_transactions(
    input_files: Optional[List[str]] = None,
    input_glob: Optional[str] = None,
    account_id: str = "default",
    model: str = "gemini-2.0-flash-lite",
    output_dir: str = "data/processed"
) -> Dict[str, pd.DataFrame]:
    """
    Step 1: Process and generalize transactions from CSV files
    
    Args:
        input_files: List of specific CSV file paths to process
        input_glob: Glob pattern to find CSV files (used if input_files is None)
        account_id: Account identifier for transactions
        model: LLM model to use for header mapping
        output_dir: Directory to save output files
        
    Returns:
        Dict containing:
        - 'clean_df': Clean transactions DataFrame
        - 'bad_df': Quarantined transactions DataFrame  
        - 'filtered_df': Filtered out transactions DataFrame
        - 'file_paths': List of processed file paths
    """
    load_dotenv()
    log = setup_logging()
    
    log.info("=" * 60)
    log.info("STEP 1: TRANSACTION GENERALIZATION")
    log.info("=" * 60)
    
    # Determine input files
    if input_files is None:
        if input_glob is None:
            input_glob = os.getenv("INPUT_GLOB", "data/raw/*.csv")
        files = sorted(glob.glob(input_glob))
    else:
        files = input_files
    
    if not files:
        log.error(f"No files found to process")
        return {
            'clean_df': pd.DataFrame(),
            'bad_df': pd.DataFrame(), 
            'filtered_df': pd.DataFrame(),
            'file_paths': []
        }
    
    log.info(f"Processing {len(files)} file(s) with model={model}")
    
    try:
        # Process files using existing ingest function
        clean_df, bad_df, filtered_df = ingest_files(files, account_id=account_id, model_name=model)
        
        # Sort DataFrames by date
        if not clean_df.empty and 'txn_date' in clean_df.columns:
            clean_df = clean_df.sort_values('txn_date', ascending=True).reset_index(drop=True)
        if not bad_df.empty and 'txn_date' in bad_df.columns:
            bad_df = bad_df.sort_values('txn_date', ascending=True).reset_index(drop=True)
        if not filtered_df.empty and 'txn_date' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('txn_date', ascending=True).reset_index(drop=True)
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        
        if not clean_df.empty:
            clean_file = os.path.join(output_dir, "clean.csv")
            write_csv(clean_df, clean_file)
            log.info(f"✓ Clean transactions: {clean_file} ({len(clean_df)} rows)")
        
        if not bad_df.empty:
            quarantine_file = os.path.join(output_dir, "quarantine.csv")
            write_csv(bad_df, quarantine_file)
            log.info(f"✓ Quarantine transactions: {quarantine_file} ({len(bad_df)} rows)")
        
        if not filtered_df.empty:
            filtered_file = os.path.join(output_dir, "filtered.csv")
            write_csv(filtered_df, filtered_file)
            log.info(f"✓ Filtered transactions: {filtered_file} ({len(filtered_df)} rows)")
        
        log.info(f"✓ Step 1 completed: {len(clean_df)} clean transactions processed")
        
        return {
            'clean_df': clean_df,
            'bad_df': bad_df,
            'filtered_df': filtered_df,
            'file_paths': files
        }
        
    except Exception as e:
        log.error(f"Step 1 failed: {e}")
        return {
            'clean_df': pd.DataFrame(),
            'bad_df': pd.DataFrame(),
            'filtered_df': pd.DataFrame(), 
            'file_paths': files
        }

def categorize_transactions(
    transactions_df: pd.DataFrame,
    taxonomy_path: str = "core/categorization/taxonomy.yaml",
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Step 2: Categorize transactions using AI classification
    
    Args:
        transactions_df: DataFrame of clean transactions to categorize
        taxonomy_path: Path to taxonomy YAML file
        output_file: Optional file path to save categorized results
        
    Returns:
        DataFrame with categorized transactions including category, subcategory, confidence, method, reasoning
    """
    load_dotenv()
    log = setup_logging()
    
    log.info("=" * 60)
    log.info("STEP 2: TRANSACTION CATEGORIZATION")
    log.info("=" * 60)
    
    if transactions_df.empty:
        log.warning("No transactions to categorize")
        return pd.DataFrame()
    
    log.info(f"Categorizing {len(transactions_df)} transactions...")
    
    # Initialize classifier
    try:
        classifier = EnsembleClassifier(taxonomy_path)
        log.info("✓ Classifier initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize classifier: {e}")
        return pd.DataFrame()
    
    # Convert DataFrame to list of dictionaries for batch processing
    transactions_list = []
    for idx, row in transactions_df.iterrows():
        transactions_list.append({
            'description': str(row.get('description', '')),
            'amount': float(row.get('amount', 0.0)),
            'card_name': str(row.get('source_bank', '')),
            'date': str(row.get('txn_date', '')),
            'original_row': row.to_dict()
        })
    
    log.info(f"Processing {len(transactions_list)} transactions with efficient batch processing...")
    
    try:
        # Use efficient batch processing
        categorized_transactions = classifier.classify_batch_efficient(transactions_list)
        
        # Convert back to DataFrame format
        final_categorized_transactions = []
        for result in categorized_transactions:
            final_categorized_transactions.append({
                **result['original_row'],
                'category': result['category'],
                'subcategory': result['subcategory'],
                'confidence': result['confidence'],
                'method': result['method'],
                'reasoning': result['reasoning']
            })
        
        # Create categorized DataFrame
        categorized_df = pd.DataFrame(final_categorized_transactions)
        
        # Sort by date first, then category and subcategory
        if 'txn_date' in categorized_df.columns:
            categorized_df = categorized_df.sort_values(['txn_date', 'category', 'subcategory'], ascending=[True, True, True])
        
        # Save categorized data if output file specified
        if output_file:
            try:
                write_csv(categorized_df, output_file)
                log.info(f"✓ Classified transactions: {output_file} ({len(categorized_df)} rows)")
            except Exception as e:
                log.error(f"Failed to save categorized data: {e}")
        
        log.info(f"✓ Step 2 completed: {len(categorized_df)} transactions categorized")
        
        return categorized_df
        
    except Exception as e:
        log.error(f"Step 2 failed: {e}")
        return pd.DataFrame()

def run_complete_pipeline(
    input_files: Optional[List[str]] = None,
    input_glob: Optional[str] = None,
    account_id: str = "default",
    model: str = "gemini-2.0-flash-lite",
    taxonomy_path: str = "core/categorization/taxonomy.yaml",
    output_dir: str = "data/processed",
    classified_output: str = "data/classified/classified.csv"
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete transaction processing pipeline (generalize + categorize)
    
    Args:
        input_files: List of specific CSV file paths to process
        input_glob: Glob pattern to find CSV files (used if input_files is None)
        account_id: Account identifier for transactions
        model: LLM model to use for header mapping
        taxonomy_path: Path to taxonomy YAML file
        output_dir: Directory to save intermediate files
        classified_output: Path to save final classified results
        
    Returns:
        Dict containing all DataFrames from both steps
    """
    load_dotenv()
    log = setup_logging()
    
    log.info("=" * 60)
    log.info("AGENTIC RAG SPENDING COACH - COMPLETE PIPELINE")
    log.info("=" * 60)
    
    # Step 1: Generalize transactions
    step1_results = generalize_transactions(
        input_files=input_files,
        input_glob=input_glob,
        account_id=account_id,
        model=model,
        output_dir=output_dir
    )
    
    if step1_results['clean_df'].empty:
        log.error("No clean transactions from Step 1, cannot proceed to Step 2")
        return step1_results
    
    # Step 2: Categorize transactions
    categorized_df = categorize_transactions(
        transactions_df=step1_results['clean_df'],
        taxonomy_path=taxonomy_path,
        output_file=classified_output
    )
    
    # Combine results
    results = {
        **step1_results,
        'categorized_df': categorized_df
    }
    
    # Generate summary
    if not categorized_df.empty:
        _log_pipeline_summary(step1_results['clean_df'], categorized_df, log)
    
    return results

def get_transaction_summary(transactions_df: pd.DataFrame) -> Dict:
    """
    Generate a summary of transaction data
    
    Args:
        transactions_df: DataFrame of transactions
        
    Returns:
        Dict containing summary statistics
    """
    if transactions_df.empty:
        return {
            'total_transactions': 0,
            'unique_cards': 0,
            'date_range': None,
            'amount_range': None,
            'categories': {}
        }
    
    summary = {
        'total_transactions': len(transactions_df),
        'unique_cards': transactions_df['source_bank'].nunique() if 'source_bank' in transactions_df.columns else 0,
        'date_range': None,
        'amount_range': None,
        'categories': {}
    }
    
    # Date range
    if 'txn_date' in transactions_df.columns:
        summary['date_range'] = {
            'start': transactions_df['txn_date'].min(),
            'end': transactions_df['txn_date'].max()
        }
    
    # Amount range
    if 'amount' in transactions_df.columns:
        summary['amount_range'] = {
            'min': float(transactions_df['amount'].min()),
            'max': float(transactions_df['amount'].max()),
            'total': float(transactions_df['amount'].sum())
        }
    
    # Category distribution
    if 'category' in transactions_df.columns:
        category_counts = transactions_df['category'].value_counts()
        summary['categories'] = category_counts.to_dict()
    
    return summary

def _log_pipeline_summary(clean_df: pd.DataFrame, categorized_df: pd.DataFrame, log):
    """Log complete pipeline summary"""
    
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    
    # Step 1 summary
    log.info("Step 1 - Transaction Generalization:")
    log.info(f"  Clean transactions: {len(clean_df)}")
    log.info(f"  Unique cards: {clean_df['source_bank'].nunique()}")
    log.info(f"  Date range: {clean_df['txn_date'].min()} to {clean_df['txn_date'].max()}")
    log.info(f"  Amount range: ${clean_df['amount'].min():.2f} to ${clean_df['amount'].max():.2f}")
    
    # Step 2 summary
    log.info("\nStep 2 - Transaction Categorization:")
    
    # Method distribution
    method_counts = categorized_df['method'].value_counts()
    log.info("  Classification Methods:")
    for method, count in method_counts.items():
        percentage = (count / len(categorized_df)) * 100
        log.info(f"    {method}: {count} transactions ({percentage:.1f}%)")
    
    # Category distribution
    category_counts = categorized_df['category'].value_counts()
    log.info(f"  Top Categories:")
    for category, count in category_counts.head(8).items():
        percentage = (count / len(categorized_df)) * 100
        log.info(f"    {category}: {count} transactions ({percentage:.1f}%)")
    
    # Confidence statistics
    avg_confidence = categorized_df['confidence'].mean()
    high_confidence = len(categorized_df[categorized_df['confidence'] >= 0.8])
    low_confidence = len(categorized_df[categorized_df['confidence'] < 0.6])
    
    log.info(f"  Confidence Statistics:")
    log.info(f"    Average confidence: {avg_confidence:.2f}")
    log.info(f"    High confidence (≥0.8): {high_confidence} transactions")
    log.info(f"    Low confidence (<0.6): {low_confidence} transactions")
    
    log.info("=" * 60)
    log.info("PIPELINE COMPLETED SUCCESSFULLY!")
    log.info("=" * 60)

if __name__ == "__main__":
    # Run complete pipeline when executed directly
    run_complete_pipeline()
