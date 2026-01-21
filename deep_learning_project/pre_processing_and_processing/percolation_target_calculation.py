import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox


# -----------------------------
# 1) Percolation: random node removal, AUC of mean LCC curve
# -----------------------------
def percolation_auc_node_removal(
    G: nx.Graph,
    step: float = 0.02,
    mc_runs: int = 20,
    seed: int = 0,
    use_trapz: bool = True,
) -> tuple[float, pd.DataFrame]:
    """
    Random node-removal percolation.
    Returns:
      - auc: scalar robustness target (area under mean LCC curve)
      - curve_df: DataFrame with columns p, s_mean, s_std

    Notes:
      - s(p) normalized by original N of the input graph (after optional LCC cleanup).
      - Assumes undirected connectivity; converts to undirected if needed.
    """
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")
    if mc_runs <= 0:
        raise ValueError("mc_runs must be >= 1.")

    if G.is_directed():
        G_u = ox.convert.to_undirected(G)
    else:
        G_u = G

    nodes = list(G_u.nodes())
    N = len(nodes)
    if N == 0:
        raise ValueError("Graph has no nodes.")

    p_vals = np.arange(0.0, 1.0 + 1e-12, step)
    s_runs = np.zeros((mc_runs, len(p_vals)), dtype=float)

    for r in range(mc_runs):
        rng = np.random.default_rng(seed + r)
        perm = rng.permutation(nodes)

        for i, p in enumerate(p_vals):
            m = int(np.floor(p * N))
            if m <= 0:
                H = G_u
            elif m >= N:
                s_runs[r, i] = 0.0
                continue
            else:
                removed = set(perm[:m])
                H = nx.subgraph_view(G_u, filter_node=lambda n, rem=removed: (n not in rem))

            if H.number_of_nodes() == 0:
                s = 0.0
            else:
                lcc_size = max((len(c) for c in nx.connected_components(H)), default=0)
                s = lcc_size / N

            s_runs[r, i] = s

    s_mean = s_runs.mean(axis=0)
    s_std = s_runs.std(axis=0, ddof=1) if mc_runs > 1 else np.zeros_like(s_mean)
    auc = np.trapz(s_mean, p_vals) if use_trapz else float(np.sum(s_mean[:-1]) * step)

    curve_df = pd.DataFrame({"p": p_vals, "s_mean": s_mean, "s_std": s_std})
    return float(auc), curve_df


# -----------------------------
# 2) Graph cleanup + safe LCC
# -----------------------------
def largest_connected_component_undirected(G: nx.Graph) -> nx.Graph:
    """Convert to undirected (if needed) and return a copy of the largest connected component."""
    if G.is_directed():
        G = ox.convert.to_undirected(G)
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    lcc_nodes = max(comps, key=len)
    return G.subgraph(lcc_nodes).copy()


# -----------------------------
# 3) Feature extraction (graph-level, aggregated)
# -----------------------------
def _safe_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def extract_graph_features(
    G: nx.Graph, 
    betweenness_k: int | None = None,
    betweenness_seed: int = 0,) -> dict:
    """
    Produce a fixed-length feature vector from a (typically OSMnx) undirected graph.
    Uses only relatively tractable features (no expensive all-pairs computations).
    """
    if G.is_directed():
        G = ox.convert.to_undirected(G)

    N = G.number_of_nodes()
    M = G.number_of_edges()

    # Basic scalars
    density = nx.density(G) if N > 1 else 0.0
    avg_degree = (2 * M / N) if N > 0 else 0.0

    # Degree distribution summaries
    deg = np.array([d for _, d in G.degree()], dtype=float) if N > 0 else np.array([], dtype=float)

    # Clustering (average)
    # For OSMnx graphs, this is usually fine on these tile sizes; if it's slow, remove it.
    #avg_clustering = float(nx.average_clustering(G)) if N > 1 else 0.0
    #transitivity = float(nx.transitivity(G)) if N > 2 else 0.0

    # Edge length summaries (OSMnx typically stores 'length' on edges)
    lengths = []

    if N <= 1:
        bc = np.array([], dtype=float)
    else:
        if betweenness_k is None:
            bc_dict = nx.betweenness_centrality(G, normalized=True)
        else:
            k = int(min(max(betweenness_k, 1), N))
            bc_dict = nx.betweenness_centrality(
                G,
                k=k,
                normalized=True,
                seed=betweenness_seed,
            )
        bc = np.array(list(bc_dict.values()), dtype=float)

    for _, _, data in G.edges(data=True):
        if "length" in data and data["length"] is not None:
            try:
                lengths.append(float(data["length"]))
            except (TypeError, ValueError):
                pass
    lengths = np.array(lengths, dtype=float)

    feats = {
        # size / density
        "n_nodes": int(N),
        "n_edges": int(M),
        "density": float(density),
        "avg_degree": float(avg_degree),

        # degree distribution
        "deg_mean": float(deg.mean()) if deg.size else float("nan"),
        "deg_std": float(deg.std(ddof=1)) if deg.size > 1 else float("nan"),
        "deg_max": float(deg.max()) if deg.size else float("nan"),
        "deg_p90": _safe_percentile(deg, 90),

        # clustering
        #"avg_clustering": avg_clustering,
        #"transitivity": transitivity,

        # edge lengths
        "edge_length_mean": float(lengths.mean()) if lengths.size else float("nan"),
        "edge_length_std": float(lengths.std(ddof=1)) if lengths.size > 1 else float("nan"),
        "edge_length_p90": _safe_percentile(lengths, 90),
        "total_edge_length": float(lengths.sum()) if lengths.size else float("nan"),

        "betw_mean": float(bc.mean()) if bc.size else float("nan"),
        "betw_std": float(bc.std(ddof=1)) if bc.size > 1 else float("nan"),
        "betw_max": float(bc.max()) if bc.size else float("nan"),
        "betw_p90": _safe_percentile(bc, 90),

        # bookkeeping
        "betweenness_k": float(betweenness_k) if betweenness_k is not None else float("nan"),
        
    }
    return feats


# -----------------------------
# 4) End-to-end: load selected tiles, compute targets + features, merge to ML table
# -----------------------------
def build_ml_dataset_end_to_end(
    selected_metadata_csv: str,
    graphs_dir: str,
    out_dir: str = "ml_dataset_outputs",
    percolation_step: float = 0.02,
    percolation_mc_runs: int = 20,
    percolation_seed: int = 0,
    use_trapz: bool = True,
    save_curves: bool = False,
    betweenness_k: int | None = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads selected tile metadata, loads each GraphML, computes:
      - percolation AUC target (node removal)
      - graph-level feature vector
    Writes:
      - targets_auc.csv
      - features.csv
      - dataset_ml.csv (merged)

    Returns:
      (targets_df, features_df, dataset_df)
    """
    graphs_path = Path(graphs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(selected_metadata_csv)

    # Determine how to locate graphs:
    # Prefer graphml_path column if present, otherwise use graphs_dir + filename inference.
    has_path = "graphml_path" in sel.columns

    target_rows = []
    feature_rows = []

    if save_curves:
        curves_path = out_path / "percolation_curves"
        curves_path.mkdir(parents=True, exist_ok=True)

    for idx, row in sel.iterrows():
        # Resolve graph file path
        if has_path and isinstance(row.get("graphml_path", None), str) and row["graphml_path"]:
            graph_file = Path(row["graphml_path"])
            if not graph_file.is_file():
                # If path stored in CSV is stale, try basename under graphs_dir
                graph_file = graphs_path / Path(row["graphml_path"]).name
        else:
            raise ValueError("selected_metadata_csv must contain a usable 'graphml_path' column.")

        if not graph_file.is_file():
            raise FileNotFoundError(f"GraphML not found for row {idx}: {graph_file}")

        # Load and standardize graph
        G = ox.load_graphml(graph_file)
        G = largest_connected_component_undirected(G)

        # Compute target
        auc, curve_df = percolation_auc_node_removal(
            G,
            step=percolation_step,
            mc_runs=percolation_mc_runs,
            seed=percolation_seed + int(idx) * 10_000,  # deterministic separation per-tile
            use_trapz=use_trapz,
        )

        target_row = {
            "row_id": int(idx),
            "graph_file": graph_file.name,
            "place": row["place"] if "place" in sel.columns else None,
            "auc_target": float(auc),
            "n_nodes_after_lcc": int(G.number_of_nodes()),
            "n_edges_after_lcc": int(G.number_of_edges()),
        }
        target_rows.append(target_row)

        if save_curves:
            curve_out = curves_path / f"{graph_file.stem}_curve.csv"
            curve_df.to_csv(curve_out, index=False)

        # Compute features
        feats = extract_graph_features(G, betweenness_k=betweenness_k, betweenness_seed=percolation_seed + int(idx),)
        feat_row = {
            "row_id": int(idx),
            "graph_file": graph_file.name,
            "place": row["place"] if "place" in sel.columns else None,
            **feats,
        }
        feature_rows.append(feat_row)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(sel)} tiles")

    targets_df = pd.DataFrame(target_rows)
    features_df = pd.DataFrame(feature_rows)

    # Merge (inner join by row_id ensures 1:1)
    dataset_df = pd.merge(features_df, targets_df[["row_id", "auc_target"]], on="row_id", how="inner")

    # Save outputs
    targets_df.to_csv(out_path / "targets_auc.csv", index=False)
    features_df.to_csv(out_path / "features.csv", index=False)
    dataset_df.to_csv(out_path / "dataset_ml.csv", index=False)

    print("\nWrote:")
    print(" -", out_path / "targets_auc.csv")
    print(" -", out_path / "features.csv")
    print(" -", out_path / "dataset_ml.csv")
    if save_curves:
        print(" - curves in:", out_path / "percolation_curves")

    return targets_df, features_df, dataset_df


# -----------------------------
# Run it
# -----------------------------
if __name__ == "__main__":
    # - selected_metadata_csv: CSV produced by your balanced selector
    # - graphs_dir: directory containing the selected 200 GraphML files (if graphml_path in CSV is stale, this is used)
    selected_metadata_csv = "tile_graphs_europe_pool/tile_metadata_selected_200_balanced.csv"
    graphs_dir = "tile_graphs_europe_selected_200_balanced"

    build_ml_dataset_end_to_end(
        selected_metadata_csv=selected_metadata_csv,
        graphs_dir=graphs_dir,
        out_dir="ml_dataset_outputs",
        percolation_step=0.02,
        percolation_mc_runs=20,
        percolation_seed=0,
        use_trapz=False,
        save_curves=True,   # set True if you want curve CSVs per tile (more disk)
        betweenness_k=200,
    )
