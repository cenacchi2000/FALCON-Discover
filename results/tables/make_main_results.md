# Table 1 — Main multi-dataset result

| dataset         | best_prior      | best_family         |   prior_auroc |   family_auroc |   prior_cap20 |   family_cap20 |   delta_auroc |   delta_cap20 |
|:----------------|:----------------|:--------------------|--------------:|---------------:|--------------:|---------------:|--------------:|--------------:|
| Adult           | trustscore      | learned discrepancy |         0.525 |          0.847 |         0.108 |          0.728 |         0.322 |         0.621 |
| Bank Marketing  | trustscore      | learned discrepancy |         0.629 |          0.859 |         0.212 |          0.74  |         0.23  |         0.528 |
| MiniBooNE       | trustscore      | learned discrepancy |         0.649 |          0.832 |         0.314 |          0.669 |         0.182 |         0.355 |
| Magic Telescope | beta scaled     | learned discrepancy |         0.566 |          0.643 |         0.075 |          0.221 |         0.077 |         0.145 |
| Nomao           | trustscore      | stability           |         0.798 |          0.835 |         0.654 |          0.728 |         0.038 |         0.074 |
| Spambase        | trustscore      | stability           |         0.691 |          0.762 |         0.468 |          0.437 |         0.071 |        -0.031 |
| Phoneme         | isotonic scaled | learned discrepancy |         0.736 |          0.7   |         0.511 |          0.333 |        -0.036 |        -0.178 |
