import osmnx as ox
import networkx as nx
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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


cities = ["Paris, France", "Moscow, Russia", "London, England"]
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
#plot_street_network(graph, place_name) idk doesnt work properly
plot_degree_distribution(results)
plot_edge_length_distribution(results)
plot_node_density(results)

# print("Plotting the city, please wait...")
# plot_clustering_coefficient(graph, place_name)
