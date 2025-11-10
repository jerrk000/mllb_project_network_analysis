import osmnx as ox
import networkx as nx
import numpy as np
from networkx.classes import Graph
from scipy import stats
import matplotlib.pyplot as plt
import random

ox.settings.use_cache = True
ox.settings.cache_folder = "./osmnx_cache"

# Define cities to analyze
#cities = ["Paris, France", "New York City, New York, USA", "Beijing, China"]
cities = ["Paris, France", "Moscow, Russia", "London, England"]

def percolation_analysis(graph, num_steps=20, removal_fraction=0.05, is_random=True):
    """
    Perform percolation analysis on a graph by removing nodes in steps and measuring LCC size.
    
    :param graph: A NetworkX graph (should be strongly connected)
    :param num_steps: Number of steps to remove nodes incrementally
    :param removal_fraction: Fraction of the original nodes to remove at each step
    :return: Lists of fractions of remaining nodes and corresponding LCC sizes
    """
    original_size = len(graph)
    remaining_node_fractions = []
    lcc_size_fractions = []

    graph_copy = graph.copy()  # Work on a copy to preserve the original graph

    for step in range(num_steps):
        # Fraction of nodes remaining
        remaining_fraction = len(graph_copy) / original_size
        remaining_node_fractions.append(remaining_fraction)
        
        # Get the size of the largest connected component
        largest_cc = max(nx.connected_components(graph_copy), key=len)
        lcc_size_fraction = len(largest_cc) / original_size
        lcc_size_fractions.append(lcc_size_fraction)

        num_to_remove = int(removal_fraction * original_size)  # Remove fraction of original nodes at each step
        
        if is_random:
            # Remove a random fraction of nodes (incremental removal)
            nodes_to_remove = random.sample(list(graph_copy.nodes), num_to_remove)
            graph_copy.remove_nodes_from(nodes_to_remove)
        else: 
            # Remove nodes with highest degree
            degrees = dict(graph_copy.degree)
            nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)[:num_to_remove]
            graph_copy.remove_nodes_from(nodes_to_remove)

    return remaining_node_fractions, lcc_size_fractions

def run_percolation_analysis_multiple_times(graph, num_steps=20, removal_fraction=0.05, num_runs=30, is_random=True):
    """
    Run percolation analysis multiple times and return the mean results.
    
    :param graph: A NetworkX graph (should be strongly connected)
    :param num_steps: Number of steps to remove nodes incrementally
    :param removal_fraction: Fraction of the original nodes to remove at each step
    :param num_runs: Number of times to run the analysis
    :return: Mean fractions of remaining nodes and mean LCC sizes over all runs
    """
    all_remaining_node_fractions = []
    all_lcc_size_fractions = []

    for _ in range(num_runs):
        remaining_node_fractions, lcc_size_fractions = percolation_analysis(
            graph, num_steps=num_steps, removal_fraction=removal_fraction, is_random=is_random
        )
        all_remaining_node_fractions.append(remaining_node_fractions)
        all_lcc_size_fractions.append(lcc_size_fractions)
    
    # Compute the mean across all runs
    mean_remaining_node_fractions = np.mean(all_remaining_node_fractions, axis=0)
    mean_lcc_size_fractions = np.mean(all_lcc_size_fractions, axis=0)

    return mean_remaining_node_fractions, mean_lcc_size_fractions

def plot_city_network(g, city_name) -> None:
    # simplified_graph = ox.simplify_graph(g)

    fig, ax = ox.plot_graph(
        g,
        node_size=0,
        edge_color="white",
        edge_linewidth=0.5,
        bgcolor="#212121",
        show=False,
        close=False,
        figsize=(20, 20),
    )

    plt.show()

# Dictionary to store results for each city
results = {}
results2 = {}

# Run percolation analysis for each city
for city in cities:
    print(f"Downloading and processing network for {city}...")
    graph = ox.graph_from_place(city, network_type='drive')
    graph = ox.convert.to_undirected(graph)

    plot_city_network(graph, city)

    # Find the largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    # Create a subgraph induced by the largest connected component
    largest_subgraph = graph.subgraph(largest_cc).copy()
    print(f"Run random percolation for {city}...")
    remaining_fractions, lcc_fractions = run_percolation_analysis_multiple_times(largest_subgraph)
    results[city] = (remaining_fractions, lcc_fractions)
    print(f"Run targeted percolation for {city}...")
    remaining_fractions2, lcc_fractions2 = run_percolation_analysis_multiple_times(largest_subgraph, is_random=False)
    results2[city] = (remaining_fractions2, lcc_fractions2)


# Plot results for all cities
print(f"Plotting...")
plt.figure(figsize=(10, 6))

for city, (remaining_fractions, lcc_fractions) in results.items():
    plt.plot(remaining_fractions, lcc_fractions, marker='o', linestyle='--', label=city)

plt.title("Percolation Analysis with random Percolation for Multiple Cities")
plt.xlabel("Fraction of Remaining Nodes")
plt.ylabel("Fraction of Original LCC Size")
plt.legend()
plt.grid(True)
plt.show()

# Plot results for all cities
plt.figure(figsize=(10, 6))

for city, (remaining_fractions, lcc_fractions) in results2.items():
    plt.plot(remaining_fractions, lcc_fractions, marker='o', linestyle='--', label=city)

plt.title("Percolation Analysis with targeted Percolation for Multiple Cities")
plt.xlabel("Fraction of Remaining Nodes")
plt.ylabel("Fraction of Original LCC Size")
plt.legend()
plt.grid(True)
plt.show()



