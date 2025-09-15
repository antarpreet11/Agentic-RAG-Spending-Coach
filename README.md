# Agentic RAG Spending Coach

A Python-based spending analysis and coaching system with AI-powered transaction processing.

## Project Structure

```
server/
├── core/                    # Core business logic
│   ├── transactions/        # Transaction processing & ETL
│   ├── common/              # Shared utilities
│   └── config.py           # Configuration
├── steps/                   # Processing pipelines
│   └── step1_transactions.py
├── data/                    # Data storage
│   ├── raw/                 # Input CSV files
│   └── processed/           # Cleaned transaction data
├── logging_config.py        # Logging setup
├── requirements.txt         # Dependencies
└── run_step1.py            # Entry point
```

## Setup

1. Create virtual environment:
   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Google API key
   ```

4. Run transaction processing:
   ```bash
   python run_step1.py
   ```

## Environment Variables

- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `GEMINI_MODEL`: Model to use (default: gemini-2.0-flash-exp)
- `INPUT_GLOB`: CSV file pattern (default: data/raw/*.csv)
- `OUT_CLEAN`: Clean data output path
- `OUT_QUAR`: Quarantine data output path
- `ACCOUNT_ID`: Account identifier (default: default)

## Features

- **Transaction Processing**: Normalize CSV data from different banks
- **AI-Powered Header Mapping**: Automatically detect column mappings
- **Data Validation**: Ensure data quality and consistency
- **Caching**: Reduce API costs with intelligent caching
- **Modular Design**: Easy to extend with new processing steps
