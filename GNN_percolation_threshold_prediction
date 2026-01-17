import osmnx as ox
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



ox.settings.use_cache = True
ox.settings.cache_folder = "./osmnx_cache"


#################### 1) Get city data ##################################################################################

cities = [
    "Amsterdam, Netherlands",
    "Paris, France",
    "London, UK",
    "Tokyo, Japan",
    "New York City, USA",
    "Singapore",
    "Barcelona, Spain",
    "Rome, Italy"
]
graphs = [ox.graph_from_place(city, network_type='drive') for city in cities]


#################### 2) Compute graph metrics ###########################################################################

def compute_graph_features(G):
    G = nx.Graph(G)
    features = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': np.mean([deg for _, deg in G.degree()]),
        'avg_clustering': nx.average_clustering(G),
        'density': nx.density(G),
        'assortativity': nx.degree_assortativity_coefficient(G),
    }
    return features


############### 3) Estimate Percolation Threshold ################################################################################

def estimate_percolation_threshold(G, step=0.02):
    G = G.copy()
    num_edges = G.number_of_edges()
    base_size = len(max(nx.connected_components(G), key=len))
    fractions = np.arange(0, 1, step)
    
    for f in fractions:
        # remove random edges
        remove_count = int(f * num_edges)
        edges_to_remove = np.random.choice(list(G.edges()), remove_count, replace=False)
        G_temp = G.copy()
        G_temp.remove_edges_from(edges_to_remove)
        
        largest_cc = len(max(nx.connected_components(G_temp), key=len))
        if largest_cc / base_size < 0.5:
            return f  # critical threshold
    
    return 1.0  # fully collapsed only at 100%

########## Get data by using above function ####################################################################
data = []
for city, G in zip(cities, graphs):
    p_c = estimate_percolation_threshold(G)
    data.append({'city': city, 'p_c': p_c, **compute_graph_features(G)})


########### Graph data preparation for GNN ########################################

def graph_to_data(G, p_c):
    # Convert NetworkX → PyG
    G = nx.Graph(G)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    
    # Compute node features
    degrees = torch.tensor([G.degree(n) for n in G.nodes()], dtype=torch.float).view(-1, 1)
    clustering = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float).view(-1, 1)
    
    x = torch.cat([degrees, clustering], dim=1)
    y = torch.tensor([p_c], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

pyg_graphs = [graph_to_data(G, estimate_percolation_threshold(G)) for G in graphs]


########################## Model: Graph NN #############################################################

class PercolationPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = GINConv(nn.Linear(in_channels, hidden_dim))
        self.conv2 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    

##################################### Training loop ##########################################

# Example split
train_loader = DataLoader(pyg_graphs[:6], batch_size=1, shuffle=True)
test_loader = DataLoader(pyg_graphs[6:], batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PercolationPredictor(in_channels=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(200):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Train loss {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds, targets = [], []
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds.append(out.item())
        targets.append(batch.y.item())

print("Predicted vs Actual p_c:")
for p, t in zip(preds, targets):
    print(f"{p:.3f} vs {t:.3f}")

########################### Evaluation and Visualisation ######################################################################

# Convert to numpy
preds = np.array(preds)
targets = np.array(targets)

# -------------------------------
# 1️ Regression metrics
# -------------------------------
mse = mean_squared_error(targets, preds)
mae = mean_absolute_error(targets, preds)
r2 = r2_score(targets, preds)

print("\n--- Evaluation Metrics ---")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# -------------------------------
# 2️ Predicted vs True scatter
# -------------------------------
plt.figure(figsize=(6,6))
sns.scatterplot(x=targets, y=preds, s=70)
plt.plot([0,1], [0,1], 'r--', label="Ideal fit")
plt.xlabel("True percolation threshold $p_c$")
plt.ylabel("Predicted $p_c$")
plt.title("Predicted vs True Percolation Thresholds")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -------------------------------
# 3️ Error distribution
# -------------------------------
errors = preds - targets
plt.figure(figsize=(6,4))
sns.histplot(errors, kde=True, bins=10, color='purple')
plt.axvline(0, color='red', linestyle='--')
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction error (pred - true)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

