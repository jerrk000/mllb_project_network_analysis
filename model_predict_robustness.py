import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------------
# 1. Download a city network
# -------------------------------
city_name = "Amsterdam, Netherlands"
G = ox.graph_from_place(city_name, network_type='drive')
G = ox.add_edge_lengths(G)

# Convert to undirected (simplifies percolation)
G = nx.Graph(G)

# -------------------------------
# 2. Compute edge-level features
# -------------------------------
edge_features = []
for u, v, data in G.edges(data=True):
    length = data.get('length', 0)
    betweenness = nx.edge_betweenness_centrality(G, normalized=True).get((u, v), 0)
    deg_u, deg_v = G.degree(u), G.degree(v)
    edge_features.append({
        'u': u, 'v': v,
        'length': length,
        'deg_u': deg_u,
        'deg_v': deg_v,
        'betweenness': betweenness
    })

df = pd.DataFrame(edge_features)

# -------------------------------
# 3. Simulate percolation
# -------------------------------
def simulate_percolation(G, fraction=0.2):
    """Randomly remove a fraction of edges and return size of largest component."""
    G_copy = G.copy()
    n_remove = int(fraction * G_copy.number_of_edges())
    edges_to_remove = np.random.choice(G_copy.number_of_edges(), n_remove, replace=False)
    for idx in edges_to_remove:
        edge = list(G_copy.edges())[idx]
        G_copy.remove_edge(*edge)
    largest_cc = max(nx.connected_components(G_copy), key=len)
    robustness = len(largest_cc) / G_copy.number_of_nodes()
    return robustness

# -------------------------------
# 4. Generate robustness labels
# -------------------------------
# For simplicity, we’ll compute one robustness value per edge by removing it
robustness_scores = []
for u, v in G.edges():
    G_temp = G.copy()
    if G_temp.has_edge(u, v):
        G_temp.remove_edge(u, v)
    largest_cc = max(nx.connected_components(G_temp), key=len)
    robustness = len(largest_cc) / G_temp.number_of_nodes()
    robustness_scores.append(robustness)

df['robustness'] = robustness_scores

# -------------------------------
# 5. Train a model to predict robustness
# -------------------------------
X = df[['length', 'deg_u', 'deg_v', 'betweenness']]
y = df['robustness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R² score: {r2_score(y_test, y_pred):.3f}")

# -------------------------------
# 6. Inspect feature importance
# -------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature importance:")
print(importances.sort_values(ascending=False))
