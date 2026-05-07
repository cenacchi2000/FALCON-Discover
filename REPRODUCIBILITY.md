# Reproducibility Guide

This artifact is organized around **table-level reproduction**. Each headline empirical object has:
1. one precomputed CSV in `data/precomputed/`;
2. one table-generation script in `scripts/`;
3. one corresponding exported markdown file in `results/tables/` once regenerated.

## Expected workflow

```bash
pip install -e . -r requirements.txt
python scripts/make_main_results.py
python scripts/make_operational_impact.py
python scripts/make_family_ablation.py
python scripts/make_extended_validation.py
python scripts/make_consequences_baselines.py
python scripts/make_uncertainty_table.py
python scripts/make_null_concentration.py
python scripts/make_figures.py
```

## Output locations

- `results/tables/*.md`
- `results/figures/*.png`
- `results/demo_summary.json`

## Notes

The bundled supplementary validation runner in `tools/` expects the user’s full paper-ready core script and is included as a review-facing interface, not as a fully self-contained public benchmark package.
