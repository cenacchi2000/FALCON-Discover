import numpy as np
from falcon_review_artifact.metrics import false_confidence_event, capture_at_alpha, falseconf_auroc
from falcon_review_artifact.scores import stability_score, analytic_discrepancy_score


def test_false_confidence_event():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0, 0, 1])
    conf = np.array([0.95, 0.93, 0.70])
    out = false_confidence_event(y_true, y_pred, conf, tau=0.90)
    assert out.tolist() == [1, 0, 0]


def test_capture_bounds():
    event = np.array([1, 0, 1, 0, 0])
    score = np.array([0.9, 0.4, 0.8, 0.1, 0.2])
    val = capture_at_alpha(event, score, alpha=0.4)
    assert 0.0 <= val <= 1.0


def test_falseconf_auroc_bounds():
    event = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.4, 0.3])
    val = falseconf_auroc(event, score)
    assert 0.0 <= val <= 1.0


def test_scores_run():
    v = stability_score(np.array([0.2]), np.array([0.8]))
    d = analytic_discrepancy_score(
        np.array([0.9]), np.array([0.2]), np.array([0.3]),
        np.array([0.7]), np.array([0.6]), np.array([0.5])
    )
    assert v.shape == (1,)
    assert d.shape == (1,)
