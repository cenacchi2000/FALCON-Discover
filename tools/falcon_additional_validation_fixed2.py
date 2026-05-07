
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FALCON-Discover supplementary validation runner.

This script loads the user's main FALCON paper-ready script dynamically and
runs the additional empirical checks that are still missing from the paper:

1) Signal-family ablation
2) Fixed-backbone robustness
3) Uncertainty intervals
4) Perturbation-design sensitivity
5) Downstream weighted-calibration experiment
6) Region-quality quantitative summary
7) Stronger learned failure-prediction / selective-risk comparators

Default focus is on the three strongest datasets:
- openml_adult
- openml_bank_marketing
- openml_miniboone

Usage (PowerShell, one line):
python falcon_additional_validation.py --core_script .\falcon_discrepancy_atlas_paperready.py --outputs .\falcon_additional --seeds 7 13 21 42 --cv_folds 5 --use_xgb --use_catboost --bootstrap_n 300

If your core script has a different filename, change --core_script accordingly.
"""
import os
import json
import math
import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_core_module(core_script_path: str):
    core_script_path = os.path.abspath(core_script_path)
    if not os.path.exists(core_script_path):
        raise FileNotFoundError(f"Core script not found: {core_script_path}")
    spec = importlib.util.spec_from_file_location("falcon_core", core_script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def safe_auroc(y_true, score):
    try:
        return float(roc_auc_score(np.asarray(y_true, dtype=int), np.asarray(score, dtype=np.float64)))
    except Exception:
        return float("nan")


def safe_auprc(y_true, score):
    try:
        return float(average_precision_score(np.asarray(y_true, dtype=int), np.asarray(score, dtype=np.float64)))
    except Exception:
        return float("nan")


def ece_binary(y_true, p, n_bins: int = 15):
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if np.any(mask):
            acc = float(np.mean(correct[mask]))
            c = float(np.mean(conf[mask]))
            ece += (np.sum(mask) / n) * abs(acc - c)
    return float(ece)


def nll_binary(y_true, p):
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_binary(y_true, p):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y_true) ** 2))


def capture_at_fraction(event, score, frac):
    event = np.asarray(event, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    n = len(event)
    k = max(1, int(math.ceil(frac * n)))
    order = np.argsort(-score)
    top = order[:k]
    total = int(np.sum(event))
    if total == 0:
        return 0.0
    return float(np.sum(event[top]) / total)


def selective_aurc(error, score):
    """
    Lower is better.
    Sort by descending score = more trusted predictions retained first.
    Compute area under risk-coverage curve.
    """
    error = np.asarray(error, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score)
    err_sorted = error[order]
    coverages = np.arange(1, len(error) + 1) / len(error)
    risks = np.cumsum(err_sorted) / np.arange(1, len(error) + 1)
    return float(np.trapz(risks, coverages))


def fit_weighted_logistic_score(X_train, y_train, X_test, sample_weight=None, C=1.0, max_iter=2500):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.asarray(X_train, dtype=np.float64))
    Xte = scaler.transform(np.asarray(X_test, dtype=np.float64))
    lr = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
    lr.fit(Xtr, np.asarray(y_train, dtype=int), sample_weight=sample_weight)
    return lr.predict_proba(Xte)[:, 1], scaler, lr



def _call_with_supported_kwargs(fn, /, *args, **kwargs):
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **accepted)


def _candidate_models_compat(core, seed, use_xgb, use_catboost, model_n_jobs):
    return _call_with_supported_kwargs(
        core.candidate_models,
        seed, use_xgb, use_catboost,
        model_n_jobs=model_n_jobs,
    )


def _fit_best_base_model_compat(core, X_train, y_train, X_val, y_val, X_test, *, seed, cv_folds, use_xgb, use_catboost, model_n_jobs):
    sig = inspect.signature(core.fit_best_base_model)
    params = list(sig.parameters.keys())

    # Handle both common core-script variants:
    # 1) fit_best_base_model(X_train, y_train, X_val, y_val, X_test, *, ...)
    # 2) fit_best_base_model(X_train, y_train, X_val, X_test, *, ...)
    positional = [X_train, y_train, X_val]
    if "y_val" in params:
        positional.append(y_val)
    positional.append(X_test)

    accepted = {
        k: v for k, v in {
            "seed": seed,
            "cv_folds": cv_folds,
            "use_xgb": use_xgb,
            "use_catboost": use_catboost,
            "model_n_jobs": model_n_jobs,
        }.items() if k in sig.parameters
    }
    return core.fit_best_base_model(*positional, **accepted)


def split_dataset(X, y_np, seed):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_np, test_size=0.20, random_state=seed, stratify=y_np
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=seed + 17, stratify=y_trainval
    )
    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        np.asarray(y_train, dtype=int),
        np.asarray(y_val, dtype=int),
        np.asarray(y_test, dtype=int),
    )


def fit_fixed_backbone_with_oof(core, X_train, y_train, X_val, X_test, *, seed, cv_folds, model_name, use_xgb, use_catboost, model_n_jobs):
    models = _candidate_models_compat(core, seed, use_xgb, use_catboost, model_n_jobs)
    if model_name not in models:
        raise ValueError(f"Backbone {model_name} not available. Available: {list(models.keys())}")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_train), dtype=np.float64)

    for tr_idx, va_idx in skf.split(X_train, y_train):
        Xtr = X_train.iloc[tr_idx].reset_index(drop=True)
        ytr = y_train[tr_idx]
        Xva = X_train.iloc[va_idx].reset_index(drop=True)
        pre = core.make_preprocessor(Xtr)
        Xt_tr = np.asarray(pre.fit_transform(Xtr), dtype=np.float64)
        Xt_va = np.asarray(pre.transform(Xva), dtype=np.float64)
        mdl = clone(models[model_name])
        mdl.fit(Xt_tr, ytr)
        oof[va_idx] = core.predict_positive_proba(mdl, Xt_va)

    pre_full = core.make_preprocessor(X_train)
    Xt_train = np.asarray(pre_full.fit_transform(X_train), dtype=np.float64)
    Xt_val = np.asarray(pre_full.transform(X_val), dtype=np.float64)
    Xt_test = np.asarray(pre_full.transform(X_test), dtype=np.float64)
    mdl_full = clone(models[model_name])
    mdl_full.fit(Xt_train, y_train)

    return core.BaseResult(
        model_name=model_name,
        pre=pre_full,
        model=mdl_full,
        p_train_oof=oof,
        p_val=core.predict_positive_proba(mdl_full, Xt_val),
        p_test=core.predict_positive_proba(mdl_full, Xt_test),
        Xt_train=Xt_train,
        Xt_val=Xt_val,
        Xt_test=Xt_test,
    )


def build_features_for_split(core, base, X_train_raw, X_split_raw, y_train, p_train_oof, Xt_split, p_split, *, seed, k=25, mix_strengths=(0.10, 0.20, 0.30)):
    miss_split = core.raw_missing_coverage(X_split_raw)
    _, supp_split = core.build_support_coverage(base.Xt_train, Xt_split)
    la_split, lpa_split, ldc_split = core.build_local_structure(base.Xt_train, y_train, p_train_oof, Xt_split, k=k)
    pert_split = core.build_neighbor_perturbations(
        Xt_split, base.Xt_train, n_neighbors=6, n_mix=8, mix_strengths=tuple(mix_strengths), seed=seed
    )
    pfeat_split = core.perturbation_features(base.model, Xt_split, pert_split, p_split)
    Xf_split, names = core.make_feature_table(
        p_split,
        miss_split,
        supp_split,
        la_split,
        lpa_split,
        ldc_split,
        pfeat_split,
    )
    return Xf_split, names


def prepare_context(core, dataset_name, X, y, *, seed, cv_folds, use_xgb, use_catboost, model_n_jobs, threshold=0.90, fixed_backbone=None, k=25, mix_strengths=(0.10, 0.20, 0.30)):
    y_np = y.astype(int).to_numpy()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y_np, seed)

    if fixed_backbone is None:
        base = _fit_best_base_model_compat(
            core, X_train, y_train, X_val, y_val, X_test,
            seed=seed, cv_folds=cv_folds, use_xgb=use_xgb, use_catboost=use_catboost, model_n_jobs=model_n_jobs
        )
    else:
        base = fit_fixed_backbone_with_oof(
            core, X_train, y_train, X_val, X_test,
            seed=seed, cv_folds=cv_folds, model_name=fixed_backbone,
            use_xgb=use_xgb, use_catboost=use_catboost, model_n_jobs=model_n_jobs
        )

    Xf_train, names = build_features_for_split(
        core, base, X_train, X_train, y_train, base.p_train_oof, base.Xt_train, base.p_train_oof,
        seed=seed + 101, k=k, mix_strengths=mix_strengths
    )
    Xf_val, _ = build_features_for_split(
        core, base, X_train, X_val, y_train, base.p_train_oof, base.Xt_val, base.p_val,
        seed=seed + 202, k=k, mix_strengths=mix_strengths
    )
    Xf_test, _ = build_features_for_split(
        core, base, X_train, X_test, y_train, base.p_train_oof, base.Xt_test, base.p_test,
        seed=seed + 303, k=k, mix_strengths=mix_strengths
    )

    pred_train = (base.p_train_oof >= 0.5).astype(int)
    pred_val = (base.p_val >= 0.5).astype(int)
    pred_test = (base.p_test >= 0.5).astype(int)

    conf_train = core.confidence_score(base.p_train_oof)
    conf_val = core.confidence_score(base.p_val)
    conf_test = core.confidence_score(base.p_test)

    falseconf_train = ((pred_train != y_train) & (conf_train >= threshold)).astype(int)
    falseconf_val = ((pred_val != y_val) & (conf_val >= threshold)).astype(int)
    falseconf_test = ((pred_test != y_test) & (conf_test >= threshold)).astype(int)
    error_train = (pred_train != y_train).astype(int)
    error_test = (pred_test != y_test).astype(int)
    correct_test = (pred_test == y_test).astype(int)

    return {
        "dataset": dataset_name,
        "seed": seed,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "base": base,
        "Xf_train": Xf_train,
        "Xf_val": Xf_val,
        "Xf_test": Xf_test,
        "names": names,
        "falseconf_train": falseconf_train,
        "falseconf_val": falseconf_val,
        "falseconf_test": falseconf_test,
        "error_train": error_train,
        "error_test": error_test,
        "correct_test": correct_test,
        "conf_test": conf_test,
        "threshold": threshold,
    }


def subset_columns(X, names, selected):
    idx = {n: i for i, n in enumerate(names)}
    return X[:, [idx[c] for c in selected]]


CERTAINTY = ["confidence", "margin", "entropy"]
SUPPORT = ["support_coverage", "local_label_agreement", "local_pred_agreement", "local_distance_coverage"]
STABILITY = ["mean_prob_drift", "max_prob_drift", "label_consistency", "confidence_consistency", "logit_variance"]
ALL_FEATS = CERTAINTY + SUPPORT + STABILITY


def run_signal_ablation_row(ctx):
    names = ctx["names"]
    rows = []
    configs = {
        "full_model": ALL_FEATS,
        "no_certainty": SUPPORT + STABILITY,
        "no_support": CERTAINTY + STABILITY,
        "no_stability": CERTAINTY + SUPPORT,
        "certainty_only": CERTAINTY,
        "support_only": SUPPORT,
        "stability_only": STABILITY,
    }
    for setting, feats in configs.items():
        Xtr = subset_columns(ctx["Xf_train"], names, feats)
        Xte = subset_columns(ctx["Xf_test"], names, feats)
        sw = np.ones(len(ctx["falseconf_train"]), dtype=np.float64)
        sw[ctx["falseconf_train"] == 1] = 3.0
        score, _, _ = fit_weighted_logistic_score(Xtr, ctx["falseconf_train"], Xte, sample_weight=sw)
        rows.append({
            "dataset": ctx["dataset"],
            "seed": ctx["seed"],
            "setting": setting,
            "capture@20": capture_at_fraction(ctx["falseconf_test"], score, 0.20),
            "falseconf_auroc": safe_auroc(ctx["falseconf_test"], score),
        })
    return rows


def run_fixed_backbone_row(core, dataset_name, X, y, *, seed, cv_folds, use_xgb, use_catboost, model_n_jobs, threshold, backbones):
    rows = []
    for bb in backbones:
        ctx = prepare_context(
            core, dataset_name, X, y, seed=seed, cv_folds=cv_folds,
            use_xgb=use_xgb, use_catboost=use_catboost, model_n_jobs=model_n_jobs,
            threshold=threshold, fixed_backbone=bb
        )
        out = core.evaluate_threshold_for_base(
            ctx["base"], ctx["X_train"], ctx["X_val"], ctx["X_test"],
            ctx["y_train"], ctx["y_val"], ctx["y_test"],
            threshold=threshold, capture_fracs=[0.20], seed=seed, bootstrap_n=200
        )
        sr = out["summary_row"]
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "fixed_backbone": bb,
            "best_prior_mode": sr["best_prior_mode"],
            "best_family_mode": sr["best_family_mode"],
            "best_prior_capture@20": sr["best_prior_capture@20"],
            "best_family_capture@20": sr["best_family_capture@20"],
            "delta_family_minus_prior_capture@20": sr["delta_family_minus_prior_capture@20"],
            "best_prior_falseconf_auroc": sr["best_prior_falseconf_auroc"],
            "best_family_falseconf_auroc": sr["best_family_falseconf_auroc"],
            "delta_family_minus_prior_falseconf_auroc": sr["delta_family_minus_prior_falseconf_auroc"],
        })
    return rows


def run_uncertainty_row(core, ctx):
    out = core.evaluate_threshold_for_base(
        ctx["base"], ctx["X_train"], ctx["X_val"], ctx["X_test"],
        ctx["y_train"], ctx["y_val"], ctx["y_test"],
        threshold=ctx["threshold"], capture_fracs=[0.20], seed=ctx["seed"], bootstrap_n=300
    )
    rows = []
    for h in out["hypotheses"]:
        if abs(h["fraction"] - 0.20) < 1e-12 and h["comparison"].startswith("best_family_vs_"):
            rows.append({
                "dataset": ctx["dataset"],
                "seed": ctx["seed"],
                "comparison": h["comparison"],
                "delta_mean": h["delta_mean"],
                "delta_ci_lo": h["delta_ci_lo"],
                "delta_ci_hi": h["delta_ci_hi"],
                "p_le_zero": h["p_le_zero"],
            })
    return rows


def run_perturbation_sensitivity(core, dataset_name, X, y, *, seed, cv_folds, use_xgb, use_catboost, model_n_jobs, threshold, lambda_grid, k_grid):
    rows = []
    # lambda sensitivity
    for lam in lambda_grid:
        ctx = prepare_context(
            core, dataset_name, X, y, seed=seed, cv_folds=cv_folds,
            use_xgb=use_xgb, use_catboost=use_catboost, model_n_jobs=model_n_jobs,
            threshold=threshold, k=25, mix_strengths=(lam,)
        )
        score = core.learned_discrepancy_score(ctx["Xf_train"], ctx["Xf_test"], ctx["names"], ctx["falseconf_train"])
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "analysis": "lambda",
            "setting": f"lambda={lam:.2f}",
            "capture@20": capture_at_fraction(ctx["falseconf_test"], score, 0.20),
            "falseconf_auroc": safe_auroc(ctx["falseconf_test"], score),
        })
    # neighborhood sensitivity
    for k in k_grid:
        ctx = prepare_context(
            core, dataset_name, X, y, seed=seed, cv_folds=cv_folds,
            use_xgb=use_xgb, use_catboost=use_catboost, model_n_jobs=model_n_jobs,
            threshold=threshold, k=int(k), mix_strengths=(0.10, 0.20, 0.30)
        )
        score = core.learned_discrepancy_score(ctx["Xf_train"], ctx["Xf_test"], ctx["names"], ctx["falseconf_train"])
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "analysis": "k_neighbors",
            "setting": f"k={int(k)}",
            "capture@20": capture_at_fraction(ctx["falseconf_test"], score, 0.20),
            "falseconf_auroc": safe_auroc(ctx["falseconf_test"], score),
        })
    return rows


def fit_weighted_platt(p_val, y_val, sample_weight):
    X = core_logit(p_val).reshape(-1, 1)
    lr = LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs")
    lr.fit(X, y_val, sample_weight=sample_weight)
    return lr


def core_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


def run_downstream_calibration(ctx):
    idx = {n: i for i, n in enumerate(ctx["names"])}
    # unweighted platt
    platt_un = LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs")
    platt_un.fit(core_logit(ctx["base"].p_val).reshape(-1, 1), ctx["y_val"])
    p_un = platt_un.predict_proba(core_logit(ctx["base"].p_test).reshape(-1, 1))[:, 1]

    # discrepancy-aware weights on validation split
    conf_val = np.maximum(ctx["base"].p_val, 1 - ctx["base"].p_val)
    w_val = (
        1.2 * conf_val +
        1.0 * (1.0 - ctx["Xf_val"][:, idx["support_coverage"]]) +
        0.9 * (1.0 - ctx["Xf_val"][:, idx["local_label_agreement"]]) +
        1.2 * ctx["Xf_val"][:, idx["mean_prob_drift"]] +
        1.0 * (1.0 - ctx["Xf_val"][:, idx["label_consistency"]]) +
        2.0 * ctx["falseconf_val"]
    )
    w_val = np.clip(w_val, 0.5, np.quantile(w_val, 0.98))
    platt_w = LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs")
    platt_w.fit(core_logit(ctx["base"].p_val).reshape(-1, 1), ctx["y_val"], sample_weight=w_val)
    p_w = platt_w.predict_proba(core_logit(ctx["base"].p_test).reshape(-1, 1))[:, 1]

    rows = []
    for name, p in [("unweighted_platt", p_un), ("weighted_platt", p_w)]:
        pred = (p >= 0.5).astype(int)
        conf = np.maximum(p, 1 - p)
        falseconf = ((pred != ctx["y_test"]) & (conf >= ctx["threshold"])).astype(int)
        rows.append({
            "dataset": ctx["dataset"],
            "seed": ctx["seed"],
            "setting": name,
            "ece": ece_binary(ctx["y_test"], p),
            "brier": brier_binary(ctx["y_test"], p),
            "nll": nll_binary(ctx["y_test"], p),
            "falseconf_prevalence": float(np.mean(falseconf)),
            "capture@20": capture_at_fraction(falseconf, conf, 0.20),
        })
    return rows


def run_region_quality(core, ctx):
    # region construction mirrored from core
    names = ctx["names"]
    D = np.column_stack([
        core.learned_discrepancy_score(ctx["Xf_train"], ctx["Xf_test"], names, ctx["falseconf_train"]),
        ctx["Xf_test"][:, names.index("support_coverage")],
        ctx["Xf_test"][:, names.index("local_label_agreement")],
        ctx["Xf_test"][:, names.index("mean_prob_drift")],
        ctx["Xf_test"][:, names.index("label_consistency")],
        ctx["conf_test"],
    ])
    from sklearn.cluster import KMeans
    region = KMeans(n_clusters=6, random_state=ctx["seed"], n_init=20).fit_predict(StandardScaler().fit_transform(D))
    per_region = []
    for r in sorted(np.unique(region)):
        mask = region == r
        per_region.append({
            "region": int(r),
            "coverage": float(np.mean(mask)),
            "falseconf_share": float(np.sum(ctx["falseconf_test"][mask]) / max(1, np.sum(ctx["falseconf_test"]))),
            "avg_support": float(np.mean(ctx["Xf_test"][mask, names.index("support_coverage")])),
            "avg_instability": float(np.mean(ctx["Xf_test"][mask, names.index("mean_prob_drift")])),
            "correctness_rate": float(np.mean(ctx["correct_test"][mask])),
        })
    df = pd.DataFrame(per_region).sort_values("falseconf_share", ascending=False).reset_index(drop=True)
    top1 = float(df.iloc[0]["falseconf_share"]) if len(df) >= 1 else np.nan
    top2 = float(df.iloc[:2]["falseconf_share"].sum()) if len(df) >= 2 else top1
    return [{
        "dataset": ctx["dataset"],
        "seed": ctx["seed"],
        "top1_falseconf_mass": top1,
        "top2_falseconf_mass": top2,
        "top_region_mean_support": float(df.iloc[0]["avg_support"]) if len(df) >= 1 else np.nan,
        "top_region_mean_instability": float(df.iloc[0]["avg_instability"]) if len(df) >= 1 else np.nan,
        "top_region_correctness_rate": float(df.iloc[0]["correctness_rate"]) if len(df) >= 1 else np.nan,
    }]


def run_failure_baselines(core, ctx):
    names = ctx["names"]
    # learned discrepancy
    score_disc = core.learned_discrepancy_score(ctx["Xf_train"], ctx["Xf_test"], names, ctx["falseconf_train"])
    # learned failure predictor: target generic error rather than false confidence
    sw = np.ones(len(ctx["error_train"]), dtype=np.float64)
    sw[ctx["error_train"] == 1] = 2.0
    score_fail, _, _ = fit_weighted_logistic_score(ctx["Xf_train"], ctx["error_train"], ctx["Xf_test"], sample_weight=sw)

    rows = []
    rows.append({
        "dataset": ctx["dataset"],
        "seed": ctx["seed"],
        "mode": "learned_failure_predictor",
        "falseconf_auroc": safe_auroc(ctx["falseconf_test"], score_fail),
        "capture@20": capture_at_fraction(ctx["falseconf_test"], score_fail, 0.20),
        "aurc": selective_aurc(1 - ctx["correct_test"], score_fail),
    })
    rows.append({
        "dataset": ctx["dataset"],
        "seed": ctx["seed"],
        "mode": "learned_discrepancy",
        "falseconf_auroc": safe_auroc(ctx["falseconf_test"], score_disc),
        "capture@20": capture_at_fraction(ctx["falseconf_test"], score_disc, 0.20),
        "aurc": selective_aurc(1 - ctx["correct_test"], score_disc),
    })
    rows.append({
        "dataset": ctx["dataset"],
        "seed": ctx["seed"],
        "mode": "confidence_raw",
        "falseconf_auroc": safe_auroc(ctx["falseconf_test"], ctx["conf_test"]),
        "capture@20": capture_at_fraction(ctx["falseconf_test"], ctx["conf_test"], 0.20),
        "aurc": selective_aurc(1 - ctx["correct_test"], ctx["conf_test"]),
    })
    return rows


def agg_mean_std(df, by_cols, metric_cols):
    rows = []
    for keys, g in df.groupby(by_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(by_cols, keys)}
        row["n"] = int(len(g))
        for c in metric_cols:
            row[c] = float(np.mean(g[c]))
            row[f"{c}_std"] = float(np.std(g[c]))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core_script", type=str, default="./falcon_discrepancy_atlas_paperready.py")
    ap.add_argument("--outputs", type=str, default="./falcon_additional")
    ap.add_argument("--datasets", nargs="+", default=["openml_adult", "openml_bank_marketing", "openml_miniboone"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42])
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--use_xgb", action="store_true")
    ap.add_argument("--use_catboost", action="store_true")
    ap.add_argument("--model_n_jobs", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--bootstrap_n", type=int, default=300)
    ap.add_argument("--fixed_backbones", nargs="+", default=["hgb", "xgb", "catboost"])
    ap.add_argument("--lambda_grid", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--k_grid", nargs="+", type=int, default=[5, 10, 20])
    args = ap.parse_args()

    ensure_dir(args.outputs)
    core = load_core_module(args.core_script)
    max_rows = args.max_rows if args.max_rows > 0 else None

    signal_rows = []
    fixed_rows = []
    interval_rows = []
    sensitivity_rows = []
    downstream_rows = []
    region_rows = []
    failure_rows = []

    for ds in args.datasets:
        X, y, meta = core.load_dataset(ds, max_rows=max_rows, seed=42)
        for seed in args.seeds:
            core.seed_all(seed)
            ctx = prepare_context(
                core, ds, X, y, seed=seed, cv_folds=args.cv_folds,
                use_xgb=args.use_xgb, use_catboost=args.use_catboost,
                model_n_jobs=args.model_n_jobs, threshold=args.threshold
            )
            signal_rows.extend(run_signal_ablation_row(ctx))
            interval_rows.extend(run_uncertainty_row(core, ctx))
            downstream_rows.extend(run_downstream_calibration(ctx))
            region_rows.extend(run_region_quality(core, ctx))
            failure_rows.extend(run_failure_baselines(core, ctx))

        # fixed backbone and perturbation sensitivity can be run dataset-wise
        for seed in args.seeds:
            fixed_rows.extend(run_fixed_backbone_row(
                core, ds, X, y, seed=seed, cv_folds=args.cv_folds,
                use_xgb=args.use_xgb, use_catboost=args.use_catboost,
                model_n_jobs=args.model_n_jobs, threshold=args.threshold,
                backbones=args.fixed_backbones
            ))
            sensitivity_rows.extend(run_perturbation_sensitivity(
                core, ds, X, y, seed=seed, cv_folds=args.cv_folds,
                use_xgb=args.use_xgb, use_catboost=args.use_catboost,
                model_n_jobs=args.model_n_jobs, threshold=args.threshold,
                lambda_grid=args.lambda_grid, k_grid=args.k_grid
            ))

    # write seed-level CSVs
    pd.DataFrame(signal_rows).to_csv(os.path.join(args.outputs, "signal_family_ablation_seed.csv"), index=False)
    pd.DataFrame(fixed_rows).to_csv(os.path.join(args.outputs, "fixed_backbone_robustness_seed.csv"), index=False)
    pd.DataFrame(interval_rows).to_csv(os.path.join(args.outputs, "uncertainty_intervals_seed.csv"), index=False)
    pd.DataFrame(sensitivity_rows).to_csv(os.path.join(args.outputs, "perturbation_sensitivity_seed.csv"), index=False)
    pd.DataFrame(downstream_rows).to_csv(os.path.join(args.outputs, "downstream_calibration_seed.csv"), index=False)
    pd.DataFrame(region_rows).to_csv(os.path.join(args.outputs, "region_quality_seed.csv"), index=False)
    pd.DataFrame(failure_rows).to_csv(os.path.join(args.outputs, "failure_predictor_baselines_seed.csv"), index=False)

    # write aggregated CSVs
    sig_df = pd.DataFrame(signal_rows)
    if len(sig_df):
        agg_mean_std(sig_df, ["dataset", "setting"], ["capture@20", "falseconf_auroc"]).to_csv(
            os.path.join(args.outputs, "signal_family_ablation_agg.csv"), index=False
        )

    fix_df = pd.DataFrame(fixed_rows)
    if len(fix_df):
        agg_mean_std(
            fix_df,
            ["dataset", "fixed_backbone", "best_prior_mode", "best_family_mode"],
            ["best_prior_capture@20", "best_family_capture@20", "delta_family_minus_prior_capture@20",
             "best_prior_falseconf_auroc", "best_family_falseconf_auroc", "delta_family_minus_prior_falseconf_auroc"]
        ).to_csv(os.path.join(args.outputs, "fixed_backbone_robustness_agg.csv"), index=False)

    int_df = pd.DataFrame(interval_rows)
    if len(int_df):
        agg_mean_std(int_df, ["dataset", "comparison"], ["delta_mean", "delta_ci_lo", "delta_ci_hi", "p_le_zero"]).to_csv(
            os.path.join(args.outputs, "uncertainty_intervals_agg.csv"), index=False
        )

    sens_df = pd.DataFrame(sensitivity_rows)
    if len(sens_df):
        agg_mean_std(sens_df, ["dataset", "analysis", "setting"], ["capture@20", "falseconf_auroc"]).to_csv(
            os.path.join(args.outputs, "perturbation_sensitivity_agg.csv"), index=False
        )

    down_df = pd.DataFrame(downstream_rows)
    if len(down_df):
        agg_mean_std(down_df, ["dataset", "setting"], ["ece", "brier", "nll", "falseconf_prevalence", "capture@20"]).to_csv(
            os.path.join(args.outputs, "downstream_calibration_agg.csv"), index=False
        )

    reg_df = pd.DataFrame(region_rows)
    if len(reg_df):
        agg_mean_std(reg_df, ["dataset"], ["top1_falseconf_mass", "top2_falseconf_mass", "top_region_mean_support", "top_region_mean_instability", "top_region_correctness_rate"]).to_csv(
            os.path.join(args.outputs, "region_quality_agg.csv"), index=False
        )

    fail_df = pd.DataFrame(failure_rows)
    if len(fail_df):
        agg_mean_std(fail_df, ["dataset", "mode"], ["falseconf_auroc", "capture@20", "aurc"]).to_csv(
            os.path.join(args.outputs, "failure_predictor_baselines_agg.csv"), index=False
        )

    with open(os.path.join(args.outputs, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "core_script": os.path.abspath(args.core_script),
            "datasets": args.datasets,
            "seeds": args.seeds,
            "cv_folds": args.cv_folds,
            "threshold": args.threshold,
            "fixed_backbones": args.fixed_backbones,
            "lambda_grid": args.lambda_grid,
            "k_grid": args.k_grid,
            "outputs": os.path.abspath(args.outputs),
        }, f, indent=2)

    print("Done. Wrote supplementary validation files to:", os.path.abspath(args.outputs))
    print("Key aggregated files:")
    for fn in [
        "signal_family_ablation_agg.csv",
        "fixed_backbone_robustness_agg.csv",
        "uncertainty_intervals_agg.csv",
        "perturbation_sensitivity_agg.csv",
        "downstream_calibration_agg.csv",
        "region_quality_agg.csv",
        "failure_predictor_baselines_agg.csv",
    ]:
        print(" -", fn)


if __name__ == "__main__":
    main()
