# build_pyg_dataset.py
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import torch
from torch_geometric.data import Data

def lcc_undirected(G: nx.Graph) -> nx.Graph:
    if G.is_directed():
        G = ox.convert.to_undirected(G)
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    lcc_nodes = max(comps, key=len)
    return G.subgraph(lcc_nodes).copy()


def _to_float(x, default=0.0) -> float:
    """Robust float conversion for GraphML string attributes."""
    if x is None:
        return float(default)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)

def graphml_to_pyg(G) -> Data:
    # deterministic node indexing
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # --- build edges + edge_attr(length) ---
    edges = []
    lengths = []

    # OSMnx graphs are often MultiGraphs: edges may be (u, v, key, data)
    # For simple graphs: edges are (u, v, data)
    is_multi = getattr(G, "is_multigraph", lambda: False)()

    if is_multi:
        edge_iter = G.edges(keys=True, data=True)
        for u, v, k, data in edge_iter:
            L = _to_float(data.get("length", 0.0), default=0.0)

            ui, vi = node_to_idx[u], node_to_idx[v]
            edges.append((ui, vi)); lengths.append([L])
            edges.append((vi, ui)); lengths.append([L])
    else:
        edge_iter = G.edges(data=True)
        for u, v, data in edge_iter:
            L = _to_float(data.get("length", 0.0), default=0.0)

            ui, vi = node_to_idx[u], node_to_idx[v]
            edges.append((ui, vi)); lengths.append([L])
            edges.append((vi, ui)); lengths.append([L])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # edge_attr shape: [num_edges, 1]
    edge_attr = torch.tensor(np.asarray(lengths, dtype=np.float32), dtype=torch.float)

    # --- node features (example: degree + log-degree; keep coords out unless you want them) ---
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    deg_log = np.log1p(deg)
    x = torch.tensor(np.stack([deg, deg_log], axis=1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_dataset(
    dataset_csv: str,                # e.g. ml_dataset_outputs/dataset_ml.csv from your script
    graphs_dir: str,                 # folder with graphml files
    out_pt: str = "pyg_dataset.pt",  # serialized list[Data]
):
    df = pd.read_csv(dataset_csv)
    graphs_dir = Path(graphs_dir)

    data_list = []
    for i, row in df.iterrows():
        graph_file = graphs_dir / row["graph_file"]
        G = ox.load_graphml(graph_file)
        G = lcc_undirected(G)

        data = graphml_to_pyg(G)
        data.y = torch.tensor([row["auc_target"]], dtype=torch.float)  # graph label
        data.place = str(row["place"])  # for grouped split

        data_list.append(data)

    torch.save(data_list, out_pt)
    print(f"Saved {len(data_list)} graphs to {out_pt}")

if __name__ == "__main__":
    # You already generate dataset_ml.csv in your end-to-end builder
    # (features + auc_target). We'll reuse it only for labels/metadata.
    build_dataset(
        dataset_csv="ml_dataset_outputs/dataset_ml.csv",
        graphs_dir="tile_graphs_europe_selected_200_balanced",
        out_pt="pyg_dataset.pt",
    )
