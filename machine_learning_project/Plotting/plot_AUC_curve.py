import pandas as pd
import matplotlib.pyplot as plt


def plot_percolation_curve_from_csv(
    curve_csv_path: str,
    title: str | None = None,
    show_std: bool = True,
):
    """
    Plot a percolation curve from a curve CSV created by this pipeline.

    Expected CSV columns (as produced by percolation_auc_node_removal):
      - p
      - s_mean
      - s_std   (optional if mc_runs=1, but typically present)

    Parameters
    ----------
    curve_csv_path : str
        Path to the percolation curve CSV file.
    title : str | None
        Plot title. If None, uses the filename.
    show_std : bool
        If True and 's_std' exists, adds ±1 std shading around s_mean.
    """
    df = pd.read_csv(curve_csv_path)

    required = {"p", "s_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Curve CSV is missing required columns: {sorted(missing)}")

    p = df["p"].to_numpy()
    s_mean = df["s_mean"].to_numpy()

    plt.figure()
    plt.plot(p, s_mean)

    if show_std and "s_std" in df.columns:
        s_std = df["s_std"].to_numpy()
        lower = (s_mean - s_std).clip(0.0, 1.0)
        upper = (s_mean + s_std).clip(0.0, 1.0)
        plt.fill_between(p, lower, upper, alpha=0.2)

    plt.xlabel("Removal fraction p")
    plt.ylabel("Largest connected component size s(p)")
    plt.ylim(0.0, 1.0)

    if title is None:
        title = curve_csv_path.split("/")[-1]
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.show()

plot_percolation_curve_from_csv(
    "ml_dataset_outputs/percolation_curves/tile_00012_Barcelona_curve.csv",
    title="Barcelona tile 12 - Percolation curve",
    show_std=True,
)
