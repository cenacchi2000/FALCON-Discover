# Algorithm Summary

1. Select the base classifier by out-of-fold AUROC on the training partition.
2. Refit the selected classifier on the training data.
3. Compute held-out scores and hard predictions on calibration and test splits.
4. Construct certainty features from confidence, margin, and entropy.
5. Construct local-evidence features from support and neighborhood agreement.
6. Construct stability features from support-preserving perturbations.
7. Form the discrepancy state `psi(x)`.
8. Fit a learned witness ranker on calibration labels using `FC_tau(x)`.
9. Rank test examples using the validation-selected discrepancy-family rule.
10. Report FalseConf-AUROC, Capture@alpha, recovered false-confidence counts, and region summaries.
11. Use calibration-only labels to derive `w_cal` for future calibration training.
