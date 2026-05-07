#!/usr/bin/env python
from __future__ import annotations
from _common import load, write_markdown

def main() -> None:
    df = load("main_results_table1.csv")
    write_markdown(df, "make_main_results.md", "Table 1 — Main multi-dataset result")
    print("wrote", "make_main_results.md")

if __name__ == "__main__":
    main()
