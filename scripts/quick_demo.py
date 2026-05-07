#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from falcon_review_artifact.scores import analytic_discrepancy_score, stability_score, calibration_weight


def main() -> None:
    confidence = np.array([0.96])
    mean_drift = np.array([0.27])
    max_drift = np.array([0.39])
    label_consistency = np.array([0.58])
    support = np.array([0.42])
    label_agreement = np.array([0.61])
    falseconf = np.array([1.0])

    summary = {
        "stability_score": float(stability_score(mean_drift, label_consistency)[0]),
        "analytic_discrepancy_score": float(
            analytic_discrepancy_score(confidence, mean_drift, max_drift, label_consistency, support, label_agreement)[0]
        ),
        "calibration_weight": float(
            calibration_weight(confidence, support, label_agreement, mean_drift, label_consistency, falseconf)[0]
        ),
    }

    out = Path("results/demo_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
