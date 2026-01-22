# plot_results.py
"""
Plots for GNN evaluation:

1) Scatter plot: y_true vs y_pred for the best GNN (default: GAT)
2) Boxplot: MSE per model across folds and seeds
3) City-wise error boxplots (absolute error) for EVERY model (one figure per model)
   + city error summary CSV per model.

Inputs (produced by train_eval.py):
- gnn_predictions.csv
- gnn_fold_details.csv

Usage:
  python plot_results.py --results-dir results_gnn
  python plot_results.py --results-dir results_gnn --best-model gat --max-cities 25 --min-graphs-per-city 3
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter_ytrue_ypred(df_preds: pd.DataFrame, model: str, outdir: Path):
    df = df_preds[df_preds["model"] == model].copy()
    if df.empty:
        raise ValueError(f"No rows found for model='{model}' in gnn_predictions.csv")

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("True robustness target (AUC)")
    plt.ylabel("Predicted robustness target (AUC)")
    plt.title(f"{model.upper()}: True vs Predicted")
    plt.tight_layout()

    outpath = outdir / f"scatter_ytrue_ypred_{model}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {outpath}")


def plot_mse_boxplot(df_folds: pd.DataFrame, outdir: Path):
    # Keep a stable order
    preferred = ["gcn", "sage", "gat", "gin"]
    present = [m for m in preferred if m in set(df_folds["model"].astype(str))]
    if not present:
        present = sorted(set(df_folds["model"].astype(str)))

    data = [df_folds[df_folds["model"] == m]["test_mse"].to_numpy() for m in present]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=[m.upper() for m in present], showfliers=True)
    plt.ylabel("Test MSE")
    plt.title("Test MSE Distribution Across Folds and Seeds")
    plt.tight_layout()

    outpath = outdir / "boxplot_mse_per_model.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved MSE boxplot to {outpath}")


def _select_cities(df_model: pd.DataFrame, max_cities: int, min_graphs_per_city: int):
    """
    Select cities to plot:
    - Keep only cities with at least min_graphs_per_city graphs
    - If too many cities, take the max_cities with the largest counts
    """
    counts = df_model.groupby("place").size().sort_values(ascending=False)
    counts = counts[counts >= min_graphs_per_city]

    if counts.empty:
        raise ValueError(
            f"No cities satisfy min_graphs_per_city={min_graphs_per_city}. "
            "Lower the threshold or check the 'place' column."
        )

    if len(counts) > max_cities:
        counts = counts.iloc[:max_cities]

    return list(counts.index), counts


def plot_citywise_error_boxplots_for_model(
    df_preds: pd.DataFrame,
    model: str,
    outdir: Path,
    max_cities: int,
    min_graphs_per_city: int,
):
    """
    City-wise boxplots of absolute error |y_pred - y_true| for a given model.
    Saves:
      - boxplot_citywise_abs_error_<model>.png
      - city_error_summary_<model>.csv
    """
    df = df_preds[df_preds["model"] == model].copy()
    if df.empty:
        print(f"[WARN] No rows for model='{model}', skipping city-wise plot.")
        return

    df["place"] = df["place"].astype(str)
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()

    cities, counts = _select_cities(df, max_cities=max_cities, min_graphs_per_city=min_graphs_per_city)

    box_data = [df.loc[df["place"] == c, "abs_error"].to_numpy() for c in cities]

    width = max(10, int(0.35 * len(cities)) + 6)
    plt.figure(figsize=(width, 5))
    plt.boxplot(box_data, labels=cities, showfliers=True)

    plt.ylabel("Absolute Error |y_pred - y_true|")
    plt.title(f"{model.upper()}: City-wise Absolute Error (Top {len(cities)} cities by sample count)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    outpath = outdir / f"boxplot_citywise_abs_error_{model}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved city-wise error boxplot to {outpath}")

    city_summary = (
        df[df["place"].isin(cities)]
        .groupby("place")
        .agg(
            n=("abs_error", "count"),
            abs_error_mean=("abs_error", "mean"),
            abs_error_median=("abs_error", "median"),
            abs_error_std=("abs_error", "std"),
        )
        .reset_index()
        .sort_values(by="abs_error_mean", ascending=False)
    )
    summary_path = outdir / f"city_error_summary_{model}.csv"
    city_summary.to_csv(summary_path, index=False)
    print(f"Saved city error summary to {summary_path}")


def plot_citywise_error_boxplots_all_models(
    df_preds: pd.DataFrame,
    outdir: Path,
    models: list,
    max_cities: int,
    min_graphs_per_city: int,
):
    """
    Create city-wise error boxplots for every model in `models`.
    """
    for m in models:
        plot_citywise_error_boxplots_for_model(
            df_preds=df_preds,
            model=m,
            outdir=outdir,
            max_cities=max_cities,
            min_graphs_per_city=min_graphs_per_city,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_gnn/", help="Directory containing CSV result files")
    parser.add_argument("--best-model", type=str, default="gat", help="Model for the scatter plot (default: gat)")
    parser.add_argument("--max-cities", type=int, default=25, help="Max number of cities shown per model")
    parser.add_argument("--min-graphs-per-city", type=int, default=3, help="Minimum graphs per city to include")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    preds_csv = results_dir / "gnn_predictions.csv"
    folds_csv = results_dir / "gnn_fold_details.csv"

    if not preds_csv.exists():
        raise FileNotFoundError(f"Missing {preds_csv}. Re-run train_eval.py with prediction saving enabled.")
    if not folds_csv.exists():
        raise FileNotFoundError(f"Missing {folds_csv}")

    df_preds = pd.read_csv(preds_csv)
    df_folds = pd.read_csv(folds_csv)

    # 1) Model-level MSE distribution
    plot_mse_boxplot(df_folds, results_dir)

    # 2) Scatter for best model
    plot_scatter_ytrue_ypred(df_preds, args.best_model.lower(), results_dir)

    # 3) City-wise error boxplots for every model present
    # Prefer canonical order if available
    preferred = ["gcn", "sage", "gat", "gin"]
    present_models = sorted(set(df_preds["model"].astype(str)))
    models = [m for m in preferred if m in present_models] + [m for m in present_models if m not in preferred]

    plot_citywise_error_boxplots_all_models(
        df_preds=df_preds,
        outdir=results_dir,
        models=models,
        max_cities=args.max_cities,
        min_graphs_per_city=args.min_graphs_per_city,
    )


if __name__ == "__main__":
    main()
