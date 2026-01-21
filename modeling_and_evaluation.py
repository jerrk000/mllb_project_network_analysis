"""
Modeling + evaluation module for percolation-robustness.

Enhanced with hyperparameter tuning capabilities.

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
- tuning_summary.csv                      : best hyperparameters and scores
- tuning_results_{model}.csv              : detailed tuning results per model
- plot_cv_r2.png                          : bar plot of mean CV R2 (error bars)
- plot_cv_mse.png                         : bar plot of mean CV MSE (error bars)
- plot_perm_importance_rf_top20.png       : top-20 permutation importances (RF)
- tuning_plot_{model}.png                 : tuning visualization for each model
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
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
                max_features="sqrt",
                min_samples_leaf=1,
            )),
        ]
    )

    return models


def _tune_hyperparameters(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    param_grid: dict,
    n_splits: int = 5,
    random_state: int = 0,
    n_iter: int = 20,
    search_method: str = "random"
) -> GridSearchCV:
    """
    Perform hyperparameter tuning with GroupKFold cross-validation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target vector
    groups : np.ndarray
        Group labels for GroupKFold
    model_name : str
        Name of model to tune ("Ridge" or "Random Forest")
    param_grid : dict
        Parameter grid for search
    n_splits : int
        Number of folds for GroupKFold
    random_state : int
        Random seed for reproducibility
    n_iter : int
        Number of iterations for RandomizedSearchCV
    search_method : str
        "grid" for GridSearchCV or "random" for RandomizedSearchCV
        
    Returns
    -------
    GridSearchCV or RandomizedSearchCV
        Fitted search object with best model
    """
    # Create base pipeline
    if model_name == "Ridge":
        base_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=random_state))
        ])
    elif model_name == "Random Forest":
        base_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1
            ))
        ])
    else:
        raise ValueError(f"Model {model_name} not supported for tuning. "
                        f"Supported: 'Ridge', 'Random Forest'")
    
    # Create GroupKFold object
    gkf = GroupKFold(n_splits=n_splits)
    
    # Choose search method
    if search_method == "grid":
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=gkf,
            scoring='neg_mean_squared_error',
            refit=True,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score='raise'
        )
    elif search_method == "random":
        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=gkf,
            scoring='neg_mean_squared_error',
            refit=True,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
            return_train_score=True,
            error_score='raise'
        )
    else:
        raise ValueError(f"search_method must be 'grid' or 'random', got {search_method}")
    
    # Fit search
    print(f"  Fitting {search_method} search with {len(param_grid)} parameter combinations...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, groups=groups)
    
    return search


def _get_param_grids(
    model_name: str,
    tuning_level: str = "medium"
) -> dict:
    """
    Get parameter grids for hyperparameter tuning.
    
    Parameters
    ----------
    model_name : str
        Name of model
    tuning_level : str
        "light" - quick tuning with few parameters
        "medium" - balanced tuning (default)
        "comprehensive" - extensive tuning (slow)
        
    Returns
    -------
    dict
        Parameter grid for the model
    """
    if model_name == "Ridge":
        if tuning_level == "light":
            return {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        elif tuning_level == "medium":
            return {"model__alpha": np.logspace(-3, 3, 13).tolist()}
        elif tuning_level == "comprehensive":
            return {"model__alpha": np.logspace(-4, 4, 17).tolist()}
    
    elif model_name == "Random Forest":
        if tuning_level == "light":
            return {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", "log2"]
            }
        elif tuning_level == "medium":
            return {
                "model__n_estimators": [100, 300, 500, 700],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__bootstrap": [True, False]
            }
        elif tuning_level == "comprehensive":
            return {
                "model__n_estimators": [100, 300, 500, 700, 1000],
                "model__max_depth": [None, 5, 10, 20, 30, 50],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 0.8],
                "model__bootstrap": [True, False]
            }
    
    raise ValueError(f"Unknown model_name or tuning_level: {model_name}, {tuning_level}")


def _plot_tuning_results(
    tuning_df: pd.DataFrame,
    model_name: str,
    out_path: Path,
    top_k_params: int = 4
):
    """
    Visualize hyperparameter tuning results.
    
    Parameters
    ----------
    tuning_df : pd.DataFrame
        DataFrame with cv_results_ from GridSearchCV/RandomizedSearchCV
    model_name : str
        Name of the model
    out_path : Path
        Output directory path
    top_k_params : int
        Number of top parameters to visualize
    """
    # Filter to only successful trials
    tuning_df = tuning_df[tuning_df['mean_test_score'].notna()].copy()
    
    if len(tuning_df) == 0:
        print(f"  No valid tuning results for {model_name}, skipping plot.")
        return
    
    # Identify parameter columns
    param_cols = [c for c in tuning_df.columns if c.startswith('param_')]
    
    # Plot top parameters
    n_params = min(len(param_cols), top_k_params)
    if n_params == 0:
        return
    
    # Create figure
    n_rows = int(np.ceil(n_params / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, param_col in enumerate(param_cols[:n_params]):
        ax = axes[idx]
        param_name = param_col.replace('param_', '').replace('model__', '')
        
        # Extract parameter values and scores
        param_data = tuning_df[[param_col, 'mean_test_score']].copy()
        
        # Convert to numeric if possible
        try:
            param_data[param_col] = pd.to_numeric(param_data[param_col])
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
        
        if is_numeric:
            # Scatter plot for numeric parameters
            ax.scatter(param_data[param_col], -param_data['mean_test_score'], 
                      alpha=0.6, s=50)
            
            # Add trend line
            try:
                valid_mask = param_data[param_col].notna() & param_data['mean_test_score'].notna()
                if valid_mask.sum() > 1:
                    x_vals = param_data.loc[valid_mask, param_col].values
                    y_vals = -param_data.loc[valid_mask, 'mean_test_score'].values
                    
                    # Polynomial fit
                    if len(np.unique(x_vals)) > 1:
                        z = np.polyfit(x_vals, y_vals, 2)  # Quadratic fit
                        p = np.poly1d(z)
                        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                        ax.plot(x_range, p(x_range), 'r--', alpha=0.8, linewidth=2)
            except:
                pass
            
            ax.set_xlabel(param_name, fontsize=11)
            ax.set_ylabel('CV MSE', fontsize=11)
            
            # Mark best parameter
            best_idx = tuning_df['rank_test_score'].idxmin()
            best_param = tuning_df.loc[best_idx, param_col]
            best_score = -tuning_df.loc[best_idx, 'mean_test_score']
            ax.axvline(x=best_param, color='green', linestyle=':', alpha=0.7, 
                      label=f'Best: {best_param:.3g}')
            
        else:
            # Box plot for categorical parameters
            unique_vals = param_data[param_col].dropna().unique()
            score_data = []
            labels = []
            
            for val in unique_vals:
                mask = param_data[param_col] == val
                scores = -param_data.loc[mask, 'mean_test_score']
                score_data.append(scores.values)
                labels.append(str(val))
            
            box = ax.boxplot(score_data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(score_data)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_xlabel(param_name, fontsize=11)
            ax.set_ylabel('CV MSE', fontsize=11)
            ax.tick_params(axis='x', rotation=45)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'{param_name}', fontsize=12)
    
    # Hide unused axes
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Hyperparameter Tuning: {model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = out_path / f"tuning_plot_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved tuning plot to: {fig_path}")


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
    
    FIXED VERSION: No double fitting
    """
    results = []
    unique_groups = pd.unique(groups)

    for g in unique_groups:
        test_mask = (groups == g)
        train_mask = ~test_mask

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Skip if no test data
        if X_test.shape[0] == 0:
            results.append({
                "place": g,
                "n_test": 0,
                "r2": np.nan,
                "mse": np.nan
            })
            continue
        
        # Fit model once
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        # Compute MSE (always valid for n ≥ 1)
        mse = mean_squared_error(y_test, pred)
        
        # Compute R² (needs n ≥ 2)
        r2 = np.nan
        if X_test.shape[0] >= 2:
            try:
                r2 = r2_score(y_test, pred)
                # Check if R² is undefined (e.g., constant y)
                if np.isnan(r2):
                    r2 = np.nan
            except:
                r2 = np.nan

        results.append({
            "place": g,
            "n_test": int(X_test.shape[0]),
            "r2": float(r2) if not np.isnan(r2) else np.nan,
            "mse": float(mse)
        })

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
    do_tuning: bool = True,
    tune_models: Optional[List[str]] = None,
    tuning_level: str = "medium",
    n_tuning_iter: int = 20,
    search_method: str = "auto",
) -> None:
    """
    Run modeling and evaluation with optional hyperparameter tuning.
    
    Parameters
    ----------
    dataset_csv : str
        Path to dataset CSV file
    out_dir : str
        Output directory for results
    target_col : str
        Name of target column
    group_col : str
        Name of group column for GroupKFold
    n_splits : int
        Number of splits for GroupKFold
    random_state : int
        Random seed for reproducibility
    perm_repeats : int
        Number of repeats for permutation importance
    do_tuning : bool
        Whether to perform hyperparameter tuning
    tune_models : List[str], optional
        List of models to tune. If None, tunes ['Ridge', 'Random Forest']
    tuning_level : str
        Level of tuning: "light", "medium", or "comprehensive"
    n_tuning_iter : int
        Number of iterations for RandomizedSearchCV
    search_method : str
        "grid", "random", or "auto" (chooses based on model)
    """
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        "dataset_csv": dataset_csv,
        "out_dir": str(out_dir),
        "target_col": target_col,
        "group_col": group_col,
        "n_splits": n_splits,
        "random_state": random_state,
        "perm_repeats": perm_repeats,
        "do_tuning": do_tuning,
        "tuning_level": tuning_level,
        "n_tuning_iter": n_tuning_iter,
        "search_method": search_method,
    }
    
    pd.Series(config).to_csv(out_path / "config.csv", header=False)
    print("=" * 70)
    print("MODELING AND EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Output directory: {out_path.resolve()}")
    
    # Load and prepare data
    print("\n1. LOADING DATA")
    print("-" * 40)
    df = pd.read_csv(dataset_csv)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    X, y, groups = _split_xy_groups(df, target_col=target_col, group_col=group_col)
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  Unique groups: {len(np.unique(groups))}")
    print(f"  Group sizes: {pd.Series(groups).value_counts().to_dict()}")
    
    # Hyperparameter tuning
    tuned_models = {}
    tuning_results = {}
    
    if do_tuning:
        print("\n2. HYPERPARAMETER TUNING")
        print("-" * 40)
        
        # Default models to tune
        if tune_models is None:
            tune_models = ["Ridge", "Random Forest"]
        
        for model_name in tune_models:
            print(f"\n  Tuning {model_name} ({tuning_level} level)...")
            
            # Get parameter grid
            try:
                param_grid = _get_param_grids(model_name, tuning_level)
                print(f"  Parameter grid size: {len(param_grid)}")
                for param, values in param_grid.items():
                    print(f"    {param}: {values}")
            except ValueError as e:
                print(f"  Warning: {e}. Skipping {model_name}.")
                continue
            
            # Determine search method
            if search_method == "auto":
                # Use grid search for Ridge (small grid), random for RF (large grid)
                method = "grid" if model_name == "Ridge" else "random"
            else:
                method = search_method
            
            print(f"  Using {method} search...")
            
            # Perform tuning
            try:
                search = _tune_hyperparameters(
                    X=X,
                    y=y,
                    groups=groups,
                    model_name=model_name,
                    param_grid=param_grid,
                    n_splits=n_splits,
                    random_state=random_state,
                    n_iter=n_tuning_iter,
                    search_method=method
                )
                
                # Store results
                tuning_results[model_name] = {
                    "best_params": search.best_params_,
                    "best_score": search.best_score_,
                    "cv_results": pd.DataFrame(search.cv_results_)
                }
                
                # Store tuned model
                tuned_models[model_name] = search.best_estimator_
                
                # Save detailed tuning results
                tuning_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
                tuning_file = out_path / f"tuning_results_{model_name.replace(' ', '_').lower()}.csv"
                tuning_df.to_csv(tuning_file, index=False)
                
                # Plot tuning results
                _plot_tuning_results(tuning_df, model_name, out_path)
                
                print(f"  ✓ Best parameters: {search.best_params_}")
                print(f"  ✓ Best CV score (MSE): {-search.best_score_:.6f}")
                print(f"  ✓ Saved results to: {tuning_file}")
                
            except Exception as e:
                print(f"  ✗ Error tuning {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Create final models dictionary
    print("\n3. CREATING MODELS")
    print("-" * 40)
    
    models = {}
    
    # Always include mean baseline
    models["Mean baseline"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyRegressor(strategy="mean")),
    ])
    print(f"  Mean baseline: added")
    
    # Add tuned or default models
    for model_name in ["Ridge", "Random Forest"]:
        if model_name in tuned_models:
            models[model_name] = tuned_models[model_name]
            print(f"  {model_name}: using tuned model")
        else:
            # Fall back to default model
            default_models = _make_models(random_state=random_state)
            if model_name in default_models:
                models[model_name] = default_models[model_name]
                print(f"  {model_name}: using default model (no tuning)")
    
    # Save tuning summary
    if tuning_results:
        tuning_summary = []
        for model_name, results in tuning_results.items():
            tuning_summary.append({
                "model": model_name,
                "best_score_mse": -results["best_score"],
                "best_score_rmse": np.sqrt(-results["best_score"]),
                "best_params": str(results["best_params"])
            })
        
        tuning_summary_df = pd.DataFrame(tuning_summary)
        tuning_summary_df.to_csv(out_path / "tuning_summary.csv", index=False)
        print(f"\n  Tuning summary saved to: {out_path / 'tuning_summary.csv'}")
    
    # -------------------------
    # 4) GroupKFold CV metrics
    # -------------------------
    print("\n4. CROSS-VALIDATION (GroupKFold)")
    print("-" * 40)
    
    cv_rows = []
    cv_fold_details = {}  # model -> (r2s, mses)

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        r2s, mses = _groupkfold_cv_metrics(X, y, groups, model=model, n_splits=n_splits)
        cv_fold_details[name] = (r2s, mses)

        cv_rows.append({
            "model": name,
            "cv_folds": n_splits,
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
            "r2_min": float(np.min(r2s)),
            "r2_max": float(np.max(r2s)),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
            "mse_min": float(np.min(mses)),
            "mse_max": float(np.max(mses)),
        })

    cv_summary = pd.DataFrame(cv_rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)
    cv_summary.to_csv(out_path / "cv_summary.csv", index=False)
    print(f"  CV summary saved to: {out_path / 'cv_summary.csv'}")

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
    print(f"  CV plots saved to: {out_path / 'plot_cv_r2.png'}, {out_path / 'plot_cv_mse.png'}")

    # ---------------------------------------
    # 5) City-held-out (leave-one-city-out)
    # ---------------------------------------
    print("\n5. CITY-HELD-OUT EVALUATION")
    print("-" * 40)
    
    heldout_rows = []
    for name, model in models.items():
        print(f"  Evaluating {name} (leave-one-city-out)...")
        per_city = _leave_one_group_out_metrics(X, y, groups, model=model)
        per_city["model"] = name
        heldout_rows.append(per_city)

    city_heldout = pd.concat(heldout_rows, ignore_index=True)
    city_heldout.to_csv(out_path / "city_heldout_scores.csv", index=False)
    print(f"  City-held-out results saved to: {out_path / 'city_heldout_scores.csv'}")

    # ---------------------------------------------------------
    # 6) Permutation importance (Random Forest, GroupKFold)
    # ---------------------------------------------------------
    print("\n6. PERMUTATION IMPORTANCE (Random Forest)")
    print("-" * 40)
    
    if "Random Forest" in models:
        rf = models["Random Forest"]
        gkf = GroupKFold(n_splits=n_splits)

        importances = []
        feature_names = X.columns.to_list()

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            print(f"  Computing permutation importance for fold {fold}/{n_splits}...")
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

        # Aggregate across folds: mean of means, and std across fold-means
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
        print(f"  Permutation importance saved to: {out_path / 'permutation_importance_rf.csv'}")
    else:
        print("  Random Forest not in models, skipping permutation importance.")

    # Console summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAll outputs saved to: {out_path.resolve()}")
    
    print("\nCross-validated performance (GroupKFold):")
    print(cv_summary[["model", "r2_mean", "r2_std", "mse_mean", "mse_std"]].to_string(index=False))
    
    if tuning_results:
        print("\nBest hyperparameters found:")
        for model_name, results in tuning_results.items():
            print(f"  {model_name}: {results['best_params']}")
            print(f"    Best CV MSE: {-results['best_score']:.6f}")
    
    print("\nPipeline completed successfully!")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Configuration
    dataset_csv = "ml_dataset_outputs/dataset_ml.csv"  # adjust if needed
    
    run_modeling_and_evaluation(
        dataset_csv=dataset_csv,
        out_dir="modeling_outputs_with_tuning",
        target_col="auc_target",
        group_col="place",
        n_splits=5,          # because of 10 cities, 5-fold GroupKFold is good default
        random_state=42,     # for better reproducibility
        perm_repeats=30,     # maybe increase to 50 for more stable importances?
        
        # Tuning parameters
        do_tuning=True,      # Set to False to skip tuning
        tune_models=["Ridge", "Random Forest"],  # Models to tune
        tuning_level="comprehensive",  # "light", "medium", or "comprehensive"
        n_tuning_iter=20,    # Iterations for RandomizedSearchCV
        search_method="auto",  # "grid", "random", or "auto"
    )