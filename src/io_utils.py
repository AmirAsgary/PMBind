"""
IO utilities: save/load DataFrames with parquet (preferred) or CSV fallback.
Ensures the code works on machines without pyarrow installed.
"""
import pandas as pd
from pathlib import Path
_HAS_PARQUET = False
try:
    import pyarrow
    _HAS_PARQUET = True
except ImportError:
    pass
def save_df(df: pd.DataFrame, path: Path, index: bool = False):
    """Save DataFrame as parquet if pyarrow is available, else CSV."""
    path = Path(path)
    if _HAS_PARQUET and path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    else:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=index)
        if path.suffix == ".parquet":
            print(f"  (pyarrow not available, saved as {csv_path.name})")
def load_df(path: Path) -> pd.DataFrame:
    """Load DataFrame, trying parquet first, then CSV fallback."""
    path = Path(path)
    if path.suffix == ".parquet" and _HAS_PARQUET and path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if path.exists() and _HAS_PARQUET:
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Cannot find {path} or {csv_path}")
