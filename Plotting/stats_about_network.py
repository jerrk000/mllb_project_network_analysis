import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import osmnx as ox

ox.settings.use_cache = True
ox.settings.log_console = True

place = "Madrid, Spain"

# 1) Download the drivable (car) street network for Madrid
G = ox.graph_from_place(place, network_type="drive", simplify=True)

# 2) Make the graph undirected
G = ox.convert.to_undirected(G)

# 3) Plot the network
# (node_size=0 makes the plot cleaner for city-scale networks)
def plot_network(G):
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_linewidth=0.6,
        show=True,
        close=False,
    )


def network_summary_table(G: nx.Graph) -> pd.DataFrame:
    """
    Create a summary table of key network measures suitable for percolation analysis.
    Path-based measures are computed on the largest connected component (LCC).
    """

    # Basic size measures
    N = G.number_of_nodes()
    E = G.number_of_edges()
    avg_degree = np.mean([d for _, d in G.degree()]) if N > 0 else np.nan
    density = nx.density(G)

    # Connected components
    #components = list(nx.connected_components(G))
    #num_components = len(components)
    #lcc_nodes = max(components, key=len)
    #G_lcc = G.subgraph(lcc_nodes)

    #lcc_size = G_lcc.number_of_nodes()
    #lcc_fraction = lcc_size / N if N > 0 else np.nan

    # Degree statistics
    degrees = np.array([d for _, d in G.degree()])
    degree_mean = degrees.mean() if len(degrees) > 0 else np.nan
    degree_var = degrees.var() if len(degrees) > 0 else np.nan
    degree_max = degrees.max() if len(degrees) > 0 else np.nan

    # Clustering
    #avg_clustering = nx.average_clustering(G)

    # Assortativity
    assortativity = nx.degree_assortativity_coefficient(G)

    """
    # Path-based measures (LCC only)
    if lcc_size > 1:
        avg_path_length = nx.average_shortest_path_length(G_lcc)
        diameter = nx.diameter(G_lcc)
        efficiency = nx.global_efficiency(G_lcc)
    else:
        avg_path_length = np.nan
        diameter = np.nan
        efficiency = np.nan
    """

    avg_path_length = np.nan
    diameter = np.nan
    efficiency = np.nan

    # Assemble table
    summary = {
        "Nodes (N)": N,
        "Edges (E)": E,
        "Average degree ⟨k⟩": avg_degree,
        "Density": density,
        #"Connected components": num_components,
        #"LCC size": lcc_size,
        #"LCC fraction": lcc_fraction,
        "Degree mean": degree_mean,
        "Degree variance": degree_var,
        "Max degree": degree_max,
        #"Average clustering": avg_clustering,
        "Degree assortativity": assortativity,
        "Average path length (LCC)": avg_path_length,
        "Diameter (LCC)": diameter,
        "Global efficiency (LCC)": efficiency,
    }

    return pd.DataFrame(summary, index=["Value"]).T


#plot_network(G)
summary = network_summary_table(G)
print(summary)

"""

def plot_street_network(graph, city_name):
    ox.plot_graph(graph, figsize=(10, 10), bgcolor="white", edge_color="gray", edge_linewidth=0.5, node_size=10,
                  show=True, save=False)
    plt.title(f"Street network for {city_name}")


def plot_degree_distribution(results, bins = 10):
    plt.title("Node Degree Distribution")
    plt.xlabel("Node Degree")
    plt.ylabel("Frequency")

    for city_data in results:
        node_degrees = [degree for _, degree in city_data['G'].degree() if degree < bins]
        plt.hist(node_degrees, bins=bins, label=city_data['city'])

    plt.legend('upper right')
    plt.show()


def plot_edge_length_distribution(results):
    plt.title("Edge Length Distribution")
    plt.xlabel("Edge Length (meters)")
    plt.ylabel("Frequency")

    for city_data in results:
        edge_lengths = [data['length'] for _, _, data in city_data['G'].edges(data=True) if 'length' in data and data['length'] < 400]
        plt.hist(edge_lengths, bins=20, label=city_data['city'])

    plt.legend('upper right')
    plt.show()


def plot_node_density(results, bins = 50):
    plt.title("Node Density Distribution")
    plt.xlabel("Distance from City Center")
    plt.ylabel("Frequency")
    for city_data in results:
        gdf_nodes = ox.graph_to_gdfs(graph, edges=False)
        distances = gdf_nodes["x"]**2 + gdf_nodes["y"]**2
        plt.hist(distances, bins=bins, label=city_data['city'])

    plt.legend('upper right')
    plt.show()

def plot_clustering_coefficient(graph, city_name):
    G = nx.Graph(graph)
    clustering = nx.clustering(G)
    node_colors = [clustering[node] for node in G.nodes()]
    node_sizes = [200 * clustering[node] for node in G.nodes()]

    ox.plot_graph(graph, bgcolor="white", node_color=node_colors, node_size=node_sizes,
                  edge_color="gray", edge_linewidth=0.5, figsize=(10, 10), show=True,
                  save=True, filepath=f"plot_clustering_coefficient_{city_name}.png")
    plt.title(f"Clustering Coefficient Heatmap: {city_name}")


cities = ["Madrid"] #, France", "Moscow, Russia", "London, England"]
results = []

for city in cities:
    graph = ox.graph_from_place(city, network_type='all')
    print(f"Graph created for {city}!")

    G = nx.Graph(graph)

    node_degrees = [degree for node, degree in G.degree()]
    node_degree_mean = np.mean(node_degrees)
    node_degree_median = np.median(node_degrees)

    node_degree_mode_result = stats.mode(node_degrees)
    node_degree_mode = node_degree_mode_result

    edge_lengths = np.array([data['length'] for u, v, data in G.edges(data=True) if 'length' in data])
    edge_length_mean = np.mean(edge_lengths)
    edge_length_median = np.median(edge_lengths)

    edge_length_mode_result = stats.mode(edge_lengths)
    edge_length_mode = edge_length_mode_result

    print("Calculating clustering coefficients...")
    clustering_coeffs = list(nx.clustering(G).values())
    clustering_mean = np.mean(clustering_coeffs)
    clustering_median = np.median(clustering_coeffs)

    print("Calculating mode...")
    clustering_mode_result = stats.mode(clustering_coeffs)
    clustering_mode = clustering_mode_result

    results.append({
        'city': city,
        'G': G,
        'node_degree_mean': node_degree_mean,
        'node_degree_median': node_degree_median,
        'node_degree_mode_result': node_degree_mode_result,
        'node_degree_mode': node_degree_mode,
        'edge_lengths': edge_lengths,
        'edge_length_mean': edge_length_mean,
        'edge_length_median': edge_length_median,
        'edge_length_mode_result': edge_length_mode_result,
        'edge_length_mode': edge_length_mode,
        'clustering_coeffs': clustering_coeffs,
        'clustering_mean': clustering_mean,
        'clustering_median': clustering_median,
        'clustering_mode_result': clustering_mode_result,
        'clustering_mode': clustering_mode,
    })

    # Print results
    print("Node Degree Statistics:")
    print(f"Mean: {node_degree_mean}")
    print(f"Median: {node_degree_median}")
    print(f"Mode: {node_degree_mode}")

    print("\nEdge Length Statistics:")
    print(f"Mean: {edge_length_mean} meters")
    print(f"Median: {edge_length_median} meters")
    print(f"Mode: {edge_length_mode} meters")

    print("\nClustering Coefficients Statistics:") # useless??
    print(f"Mean: {clustering_mean}")
    print(f"Median: {clustering_median}")
    print(f"Mode: {clustering_mode}")


print("Plotting infos...")
plot_street_network(graph, "Madrid") #idk doesnt work properly
plot_degree_distribution(results)
plot_edge_length_distribution(results)
plot_node_density(results)

# print("Plotting the city, please wait...")
# plot_clustering_coefficient(graph, place_name)
"""