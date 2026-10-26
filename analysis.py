"""Reproduce the report figures and numbers: python analysis.py"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import mlr  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SEED = 42


def run(key: str) -> dict:
    p = mlr.PROBLEMS[key]
    df = mlr.load_dataset(p["path"])
    prep = mlr.prepare_data(df, p["target"], p["drop"], p["group"], 0.2, SEED)

    curves, rankings = {}, {}
    for label, method in mlr.SELECTION_METHODS.items():
        rankings[method] = mlr.rank_features(prep.X_train, prep.y_train, method, SEED)
        curves[method] = mlr.k_curve(prep.X_train, prep.y_train, rankings[method], SEED)
    k = mlr.suggest_k(curves["forward"])
    selected = rankings["forward"][:k]

    result = mlr.fit_ols(prep.X_train[selected], prep.y_train)
    iv = mlr.intervals(result, prep.X_test[selected], 0.05).join(prep.y_test.rename("actual"))
    coverage = float(((iv.actual >= iv.pi_low) & (iv.actual <= iv.pi_high)).mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, curve in curves.items():
        ax.plot(curve["k"], curve["CV_RMSE"], marker="o", ms=3, label=method)
    ax.axvline(k, ls="--", c="gray")
    ax.set(xlabel="number of features k", ylabel="5-fold CV RMSE", title=f"{key}: feature selection curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{key}_feature_curve.png", dpi=150)
    plt.close(fig)

    d = iv.sort_values("pred").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(d.index, d.pi_low, d.pi_high, color="#3498db", alpha=0.18, label="95% prediction interval")
    ax.fill_between(d.index, d.ci_low, d.ci_high, color="#3498db", alpha=0.5, label="95% confidence interval")
    ax.plot(d.index, d.pred, c="#1f4e79", lw=1.5, label="prediction")
    ax.scatter(d.index, d.actual, s=6, c="#e67e22", label="actual", zorder=3)
    ax.set(xlabel="test samples (sorted by prediction)", ylabel=p["target"],
           title=f"{key}: test-set prediction (PI coverage {coverage:.1%})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{key}_prediction_interval.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(iv.actual, iv.pred, s=6, alpha=0.6)
    lim = [min(iv.actual.min(), iv.pred.min()), max(iv.actual.max(), iv.pred.max())]
    axes[0].plot(lim, lim, "--", c="gray")
    axes[0].set(xlabel="actual", ylabel="predicted", title="actual vs predicted (test)")
    axes[1].scatter(result.fittedvalues, result.resid, s=6, alpha=0.5)
    axes[1].axhline(0, ls="--", c="gray")
    axes[1].set(xlabel="fitted", ylabel="residual", title="residuals vs fitted (train)")
    stats.probplot(result.resid, dist="norm", plot=axes[2])
    axes[2].set_title("normal Q-Q (train residuals)")
    fig.tight_layout()
    fig.savefig(OUT / f"{key}_diagnostics.png", dpi=150)
    plt.close(fig)

    coef = mlr.coefficient_table(result)
    return {
        "shape": list(df.shape),
        "n_features_after_prep": prep.X_train.shape[1],
        "prep_log": prep.log,
        "suggested_k": {m: mlr.suggest_k(c) for m, c in curves.items()},
        "selected": selected,
        "train": mlr.metrics(prep.y_train, result.fittedvalues, k),
        "test": mlr.metrics(prep.y_test, iv.pred, k),
        "pi_coverage": coverage,
        "coef": coef.round(6).to_dict(orient="index"),
        "comparison": mlr.compare_models(prep, selected, SEED).round(5).to_dict(orient="records"),
        "assumptions": mlr.assumption_tests(result).round(4).to_dict(orient="records"),
        "vif": mlr.vif_table(prep.X_train[selected]).round(2).to_dict(orient="records"),
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    results = {key: run(key) for key in mlr.PROBLEMS}
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1, default=float))
