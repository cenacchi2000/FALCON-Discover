# Tables 5A–5C — Consequences and baselines

## A. Downstream calibration

| analysis          |   Adult |   Bank |   MiniBooNE |   Mean |
|:------------------|--------:|-------:|------------:|-------:|
| Unweighted ECE    |   0.006 |  0.011 |       0.004 |  0.007 |
| Weighted ECE      |   0.004 |  0.01  |       0.003 |  0.006 |
| Unweighted Brier  |   0.086 |  0.061 |       0.04  |  0.063 |
| Weighted Brier    |   0.065 |  0.051 |       0.033 |  0.058 |
| Unweighted NLL    |   0.273 |  0.195 |       0.136 |  0.202 |
| Weighted NLL      |   0.22  |  0.157 |       0.117 |  0.192 |
| Unweighted Cap@20 |   0.022 |  0.008 |       0.008 |  0.012 |
| Weighted Cap@20   |   0.024 |  0.012 |       0.009 |  0.018 |

## B. Region quality

| analysis               |   Adult |   Bank |   MiniBooNE |   Mean |
|:-----------------------|--------:|-------:|------------:|-------:|
| Top-1 FC mass          |   0.736 |  0.667 |       0.41  |  0.604 |
| Top-2 FC mass          |   0.902 |  0.842 |       0.768 |  0.837 |
| Top-region support     |   0.95  |  0.943 |       0.926 |  0.939 |
| Top-region instability |   0.008 |  0.007 |       0.022 |  0.012 |

## C. Stronger comparator baselines

| analysis                 |   Adult |   Bank |   MiniBooNE |   Mean |
|:-------------------------|--------:|-------:|------------:|-------:|
| Failure predictor AUROC  |   0.483 |  0.655 |       0.708 |  0.615 |
| Failure predictor Cap@20 |   0     |  0.063 |       0.249 |  0.104 |
| Selective / AURC         |   0.028 |  0.014 |       0.007 |  0.016 |
| Disc. family AUROC       |   0.85  |  0.858 |       0.833 |  0.847 |
| Disc. family Cap@20      |   0.744 |  0.75  |       0.676 |  0.723 |

## Full weighted calibration across all seven datasets

| dataset   |   ece_unweighted |   ece_weighted |   delta_ece |   cap20_unweighted |   cap20_weighted |   delta_cap20 |
|:----------|-----------------:|---------------:|------------:|-------------------:|-----------------:|--------------:|
| Adult     |            0.006 |          0.004 |      -0.002 |              0.022 |            0.024 |         0.002 |
| Bank      |            0.011 |          0.01  |      -0.001 |              0.008 |            0.012 |         0.004 |
| MiniBooNE |            0.004 |          0.003 |      -0.001 |              0.008 |            0.009 |         0.001 |
| Magic     |            0.014 |          0.011 |      -0.003 |              0.064 |            0.143 |         0.079 |
| Nomao     |            0.005 |          0.004 |      -0.001 |              0     |            0.002 |         0.002 |
| Spambase  |            0.015 |          0.012 |      -0.003 |              0.023 |            0.05  |         0.027 |
| Phoneme   |            0.017 |          0.014 |      -0.003 |              0.015 |            0.017 |         0.002 |
