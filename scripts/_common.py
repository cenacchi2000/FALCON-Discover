from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "precomputed"
OUT_TABLES = ROOT / "results" / "tables"
OUT_FIGS = ROOT / "results" / "figures"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def write_markdown(df: pd.DataFrame, out_name: str, title: str) -> None:
    text = f"# {title}\n\n" + df.to_markdown(index=False) + "\n"
    (OUT_TABLES / out_name).write_text(text)
