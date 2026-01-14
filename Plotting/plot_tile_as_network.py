import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt


def plot_tile_network(
    graphml_path: str,
    title: str | None = None,
    node_size: float = 0.0,
    edge_linewidth: float = 0.7,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plot a single tile network from a GraphML file.

    This uses OSMnx's plotting utilities when possible (best for street networks).
    Falls back to a plain NetworkX plot if geometry is missing.

    Parameters
    ----------
    graphml_path : str
        Path to the tile GraphML file.
    title : str | None
        Optional plot title. If None, uses the filename.
    node_size : float
        Node marker size. Set to 0 to hide nodes (recommended for dense street graphs).
    edge_linewidth : float
        Width of edges.
    show : bool
        Whether to display the plot window.
    save_path : str | None
        If provided, saves the plot to this file (e.g., "tile.png").
    """
    G = ox.load_graphml(graphml_path)

    # Ensure undirected (matches your pipeline and makes behavior consistent)
    if G.is_directed():
        G = ox.convert.to_undirected(G)

    if title is None:
        title = graphml_path.split("/")[-1]

    # Prefer OSMnx street plotting (uses geometry if present)
    try:
        fig, ax = ox.plot_graph(
            G,
            node_size=node_size,
            edge_linewidth=edge_linewidth,
            show=False,
            close=False,
        )
        ax.set_title(title)
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    except Exception:
        # Fallback: NetworkX plot using x/y if available
        pos = {}
        for n, data in G.nodes(data=True):
            if "x" in data and "y" in data:
                pos[n] = (data["x"], data["y"])

        plt.figure()
        if pos:
            nx.draw(
                G,
                pos=pos,
                node_size=node_size,
                width=edge_linewidth,
            )
        else:
            # Last resort: no positions, use spring layout (not geographic)
            nx.draw(
                G,
                node_size=node_size,
                width=edge_linewidth,
            )

        plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

plot_tile_network(
    "tile_graphs_europe_selected_200_balanced/tile_00012_Barcelona.graphml",
    title="Barcelona tile 12",
    node_size=0,
    edge_linewidth=0.8,
)
