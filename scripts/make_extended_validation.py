#!/usr/bin/env python
from __future__ import annotations
from _common import load, OUT_TABLES

def main() -> None:
    a = load("extended_validation_signal_table4A.csv")
    b = load("extended_validation_backbone_table4B.csv")
    c = load("extended_validation_perturbation_table4C.csv")
    text = "# Tables 4A–4C — Extended validation\n\n"
    text += "## A. Signal-family necessity\n\n" + a.to_markdown(index=False) + "\n\n"
    text += "## B. Fixed-backbone robustness\n\n" + b.to_markdown(index=False) + "\n\n"
    text += "## C. Perturbation sensitivity\n\n" + c.to_markdown(index=False) + "\n"
    (OUT_TABLES / "make_extended_validation.md").write_text(text)
    print("wrote make_extended_validation.md")

if __name__ == "__main__":
    main()
