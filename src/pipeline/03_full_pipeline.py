"""
Full pipeline: data processing → model training → evaluation.
Equivalent to running 01_data_pipeline.py then 02_model_pipeline.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_processing.data_engineering import merged_dataset, MERGED_OUT
from src.models.train_model import main as train_main

if __name__ == "__main__":
    print("=== Stage 1: Data Pipeline ===")
    df = merged_dataset()
    print(f"Data pipeline complete. {len(df):,} rows saved to {MERGED_OUT}\n")

    print("=== Stage 2: Model Pipeline ===")
    train_main()
    print("\nFull pipeline complete.")
