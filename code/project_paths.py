from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
# Centralized project path definitions used across scripts.
CODE_DIR = ROOT_DIR / "code"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
RESULT_DIR = ROOT_DIR / "result"
DOCS_DIR = ROOT_DIR / "docs"
