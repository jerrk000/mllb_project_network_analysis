"""
Modeling + evaluation module for percolation-robustness.

Inputs
------
- dataset_ml.csv produced earlier (features + auc_target), must include:
  - auc_target (float)
  - place (string)  -> used for GroupKFold / city-held-out evaluation
  - feature columns (numeric)

Outputs (written to out_dir)
----------------------------
- cv_summary.csv                          : cross-validated metrics summary per model
- city_heldout_scores.csv                 : per-city held-out R2/MSE per model
- permutation_importance_rf.csv           : permutation importance table (RF, grouped CV)
- plot_cv_r2.png                          : bar plot of mean CV R2 (error bars)
- plot_cv_mse.png                         : bar plot of mean CV MSE (error bars)
- plot_perm_importance_rf_top20.png       : top-20 permutation importances (RF)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance


# -----------------------------
# Utilities
# -----------------------------
def _split_xy_groups(df: pd.DataFrame, target_col: str, group_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in dataset.")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column '{group_col}' in dataset (needed for GroupKFold).")

    # Features: numeric columns except target; also drop non-feature identifiers if present
    drop_cols = {target_col}
    # keep group column for splitting but exclude from X
    drop_cols.add(group_col)

    # Common non-feature columns to drop if present
    for c in ["row_id", "graph_file", "graphml_path", "global_tile_id"]:
        if c in df.columns:
            drop_cols.add(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].to_numpy(dtype=float)
    groups = df[group_col].astype(str).to_numpy()

    # Keep only numeric features (defensive)
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after filtering.")
    return X, y, groups


def _make_models(random_state: int = 0) -> Dict[str, object]:
    """
    Baselines + Random Forest.
    - Mean baseline: DummyRegressor(strategy="mean")
    - Ridge: standardized features
    - RF: tree-based (no scaling needed, but impute missing)
    """
    models = {}

    models["Mean baseline"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )

    models["Ridge"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=600,
                random_state=random_state,
                n_jobs=-1,
                max_features="auto",   # sklearn will map appropriately; if warning, set to 1.0 or "sqrt"
                min_samples_leaf=1,
            )),
        ]
    )

    return models


def _groupkfold_cv_metrics(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    n_splits: int = 5,
) -> Tuple[List[float], List[float]]:
    """
    Returns fold-wise (R2 list, MSE list) using GroupKFold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    r2s, mses = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2s.append(r2_score(y_test, pred))
        mses.append(mean_squared_error(y_test, pred))

    return r2s, mses


def _leave_one_group_out_metrics(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model,
) -> pd.DataFrame:
    """
    City-held-out evaluation: for each unique group, train on all other groups and test on this group.
    Returns per-group R2 and MSE.
    """
    results = []
    unique_groups = pd.unique(groups)

    for g in unique_groups:
        test_mask = (groups == g)
        train_mask = ~test_mask

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Guard: need at least 2 samples in test to compute R2 meaningfully
        if X_test.shape[0] < 2:
            r2 = np.nan
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)

        results.append({"place": g, "n_test": int(X_test.shape[0]), "r2": float(r2) if r2 == r2 else np.nan, "mse": float(mse)})

    return pd.DataFrame(results).sort_values(["place"]).reset_index(drop=True)


def _plot_bar_with_error(
    labels: List[str],
    means: List[float],
    stds: List[float],
    ylabel: str,
    title: str,
    out_path: Path,
):
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_perm_importance_topk(
    perm_df: pd.DataFrame,
    top_k: int,
    title: str,
    out_path: Path,
):
    """
    perm_df columns: feature, importance_mean, importance_std
    """
    d = perm_df.sort_values("importance_mean", ascending=False).head(top_k).iloc[::-1]  # reverse for horizontal plot
    plt.figure(figsize=(8, max(4, 0.3 * len(d))))
    y = np.arange(len(d))
    plt.barh(y, d["importance_mean"].to_numpy(), xerr=d["importance_std"].to_numpy(), capsize=3)
    plt.yticks(y, d["feature"].tolist())
    plt.xlabel("Permutation importance (mean decrease in R²)")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main end-to-end runner
# -----------------------------
def run_modeling_and_evaluation(
    dataset_csv: str,
    out_dir: str = "modeling_outputs",
    target_col: str = "auc_target",
    group_col: str = "place",
    n_splits: int = 5,
    random_state: int = 0,
    perm_repeats: int = 30,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    X, y, groups = _split_xy_groups(df, target_col=target_col, group_col=group_col)

    # Models
    models = _make_models(random_state=random_state)

    # -------------------------
    # 1) GroupKFold CV metrics
    # -------------------------
    cv_rows = []
    cv_fold_details = {}  # model -> (r2s, mses)

    for name, model in models.items():
        r2s, mses = _groupkfold_cv_metrics(X, y, groups, model=model, n_splits=n_splits)
        cv_fold_details[name] = (r2s, mses)

        cv_rows.append({
            "model": name,
            "cv_folds": n_splits,
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
        })

    cv_summary = pd.DataFrame(cv_rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)
    cv_summary.to_csv(out_path / "cv_summary.csv", index=False)

    # Plots: CV R2 and MSE
    _plot_bar_with_error(
        labels=cv_summary["model"].tolist(),
        means=cv_summary["r2_mean"].tolist(),
        stds=cv_summary["r2_std"].tolist(),
        ylabel="R² (GroupKFold mean ± std)",
        title=f"Cross-validated R² (GroupKFold, n_splits={n_splits})",
        out_path=out_path / "plot_cv_r2.png",
    )
    _plot_bar_with_error(
        labels=cv_summary["model"].tolist(),
        means=cv_summary["mse_mean"].tolist(),
        stds=cv_summary["mse_std"].tolist(),
        ylabel="MSE (GroupKFold mean ± std)",
        title=f"Cross-validated MSE (GroupKFold, n_splits={n_splits})",
        out_path=out_path / "plot_cv_mse.png",
    )

    # ---------------------------------------
    # 2) City-held-out (leave-one-city-out)
    # ---------------------------------------
    heldout_rows = []
    for name, model in models.items():
        per_city = _leave_one_group_out_metrics(X, y, groups, model=model)
        per_city["model"] = name
        heldout_rows.append(per_city)

    city_heldout = pd.concat(heldout_rows, ignore_index=True)
    city_heldout.to_csv(out_path / "city_heldout_scores.csv", index=False)

    # ---------------------------------------------------------
    # 3) Permutation importance (Random Forest, GroupKFold)
    # ---------------------------------------------------------
    # We compute permutation importance fold-by-fold, then aggregate across folds.
    rf = models["Random Forest"]
    gkf = GroupKFold(n_splits=n_splits)

    importances = []
    feature_names = X.columns.to_list()

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf.fit(X_train, y_train)

        # Permutation importance with scoring='r2' on the held-out fold
        pi = permutation_importance(
            rf,
            X_test,
            y_test,
            scoring="r2",
            n_repeats=perm_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        for f_name, mean_imp, std_imp in zip(feature_names, pi.importances_mean, pi.importances_std):
            importances.append({
                "fold": fold,
                "feature": f_name,
                "importance_mean": float(mean_imp),
                "importance_std": float(std_imp),
            })

    perm_df = pd.DataFrame(importances)

    # Aggregate across folds: mean of means, and std across fold-means (more report-friendly)
    agg = (
        perm_df.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    agg.to_csv(out_path / "permutation_importance_rf.csv", index=False)

    _plot_perm_importance_topk(
        perm_df=agg,
        top_k=20,
        title=f"Permutation importance (Random Forest, GroupKFold; repeats={perm_repeats})",
        out_path=out_path / "plot_perm_importance_rf_top20.png",
    )

    # Console summary (useful when running interactively)
    print("\nSaved outputs to:", out_path.resolve())
    print("\nCross-validated summary (GroupKFold):")
    print(cv_summary)

    print("\nCity-held-out performance (head):")
    print(city_heldout.sort_values(["model", "place"]).head(15))

    print("\nTop permutation importances (RF):")
    print(agg.head(10))


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    dataset_csv = "ml_dataset_outputs/dataset_ml.csv"  # adjust if needed
    run_modeling_and_evaluation(
        dataset_csv=dataset_csv,
        out_dir="modeling_outputs",
        target_col="auc_target",
        group_col="place",
        n_splits=5,          # with 10 cities, 5-fold GroupKFold is a good default
        random_state=0,
        perm_repeats=30,     # increase to 50 for more stable importances (slower)
    )
