# models.py
"""
Graph-level regression models for benchmarking:
- GCN
- GraphSAGE
- GAT
- GIN (optionally GINE when edge_attr is provided)

Designed for PyTorch Geometric `Data` / `Batch` objects with:
  - x: [N, in_dim]
  - edge_index: [2, E]
  - batch: [N] (optional; if missing, assumed all nodes belong to one graph)
  - edge_attr: [E, 1] (optional; e.g., edge length in meters duplicated for both directions)

Notes on edge lengths:
  - GCNRegressor can optionally use edge lengths as `edge_weight` via a transform
    (default uses inverse length).
  - GINRegressor can optionally use edge lengths via GINEConv.
  - GraphSAGE and GAT in their standard PyG forms do not consume edge_attr directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    global_mean_pool,
)



PoolingType = Literal["mean"]


def _get_batch_vector(x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    """Ensure a valid batch vector exists."""
    if batch is None:
        return x.new_zeros(x.size(0), dtype=torch.long)
    return batch


class MLP(nn.Module):
    """Simple MLP with dropout and ReLU."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        last_activation: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self.last_activation = last_activation
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.last_activation:
            x = F.relu(x)
        return x


@dataclass
class ModelConfig:
    in_dim: int
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    pooling: PoolingType = "mean"
    # Head MLP
    head_hidden_dim: int = 64
    head_layers: int = 2  # including output layer
    # GAT specifics
    gat_heads: int = 4
    gat_concat: bool = True  # if True, hidden_dim is per-head dim; output dim = hidden_dim * heads
    # Edge-length handling for GCN
    gcn_edge_weight: bool = True
    edge_weight_mode: Literal["inverse_length", "negative_length", "none"] = "inverse_length"
    edge_eps: float = 1e-6
    # GIN specifics
    gin_use_edge_attr: bool = True  # if True and edge_attr present, use GINEConv; else GINConv


class GraphRegressorBase(nn.Module):
    """Base class: encoder -> message passing -> pooling -> head."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.node_encoder = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.dropout = cfg.dropout

        if cfg.pooling != "mean":
            raise ValueError(f"Unsupported pooling={cfg.pooling}")

        # regression head; input dim depends on backbone (overridden if needed)
        self.head_in_dim = cfg.hidden_dim
        self.head = MLP(
            in_dim=self.head_in_dim,
            hidden_dim=cfg.head_hidden_dim,
            out_dim=1,
            num_layers=cfg.head_layers,
            dropout=cfg.dropout,
        )

    def pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(x, batch)

    def forward(self, data) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class GCNRegressor(GraphRegressorBase):
    """
    Graph Convolutional Network for graph-level regression.

    Optional: uses edge lengths (edge_attr) as edge weights.
    Recommended default: inverse length => shorter edges contribute more.

    If edge_attr is missing or cfg.gcn_edge_weight is False, falls back to unweighted GCN.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(GCNConv(cfg.hidden_dim, cfg.hidden_dim, add_self_loops=True, normalize=True))

    def _edge_weight_from_length(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr expected shape [E, 1] or [E]
        if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
            length = edge_attr.view(-1)
        else:
            length = edge_attr.view(-1)

        eps = self.cfg.edge_eps
        mode = self.cfg.edge_weight_mode

        if mode == "inverse_length":
            w = 1.0 / (length + eps)
        elif mode == "negative_length":
            # Larger length -> smaller weight after exp; simple monotone transform.
            w = torch.exp(-length)
        elif mode == "none":
            w = None  # type: ignore
        else:
            raise ValueError(f"Unknown edge_weight_mode={mode}")

        # normalize weights for numerical stability (optional but usually helpful)
        if w is not None:
            w = w / (w.mean().clamp_min(eps))
        return w  # type: ignore

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = _get_batch_vector(x, getattr(data, "batch", None))

        x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_weight = None
        if self.cfg.gcn_edge_weight and hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_weight = self._edge_weight_from_length(data.edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = self.pool(x, batch)
        out = self.head(g).view(-1)  # [num_graphs]
        return out


class GraphSAGERegressor(GraphRegressorBase):
    """
    GraphSAGE for graph-level regression.

    Note: Standard SAGEConv does not use edge_attr. We store edge_attr in the dataset
    for potential use in other models (GCN weights, GINEConv, etc.).
    """
    def __init__(self, cfg: ModelConfig, aggr: str = "mean"):
        super().__init__(cfg)
        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(SAGEConv(cfg.hidden_dim, cfg.hidden_dim, aggr=aggr))

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = _get_batch_vector(x, getattr(data, "batch", None))

        x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = self.pool(x, batch)
        out = self.head(g).view(-1)
        return out


class GATRegressor(GraphRegressorBase):
    """
    Graph Attention Network for graph-level regression.

    Standard GATConv does not use edge_attr unless using specialized variants;
    this implementation is the canonical baseline GAT.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        heads = cfg.gat_heads
        concat = cfg.gat_concat

        # If concat=True, each layer outputs hidden_dim * heads
        # To keep a stable hidden size across layers, we set per-head dim accordingly.
        if concat:
            per_head_dim = max(1, cfg.hidden_dim // heads)
            layer_out = per_head_dim * heads
        else:
            per_head_dim = cfg.hidden_dim
            layer_out = cfg.hidden_dim

        self.node_encoder = nn.Linear(cfg.in_dim, layer_out)
        self.head_in_dim = layer_out
        self.head = MLP(
            in_dim=self.head_in_dim,
            hidden_dim=cfg.head_hidden_dim,
            out_dim=1,
            num_layers=cfg.head_layers,
            dropout=cfg.dropout,
        )

        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(
                GATConv(
                    in_channels=layer_out,
                    out_channels=per_head_dim,
                    heads=heads,
                    concat=concat,
                    dropout=cfg.dropout,
                    add_self_loops=True,
                )
            )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = _get_batch_vector(x, getattr(data, "batch", None))

        x = self.node_encoder(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = self.pool(x, batch)
        out = self.head(g).view(-1)
        return out

class GINRegressor(GraphRegressorBase):
    """
    GIN for graph-level regression (topology-only).

    This implementation uses GINConv and ignores edge_attr to ensure compatibility
    across older torch-geometric versions where GINEConv API differs.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.gin_ops = nn.ModuleList()
        for _ in range(cfg.num_layers):
            nn_mlp = MLP(
                in_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                out_dim=cfg.hidden_dim,
                num_layers=2,
                dropout=cfg.dropout,
            )
            self.gin_ops.append(GINConv(nn_mlp, train_eps=True))

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = _get_batch_vector(x, getattr(data, "batch", None))

        x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.gin_ops:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = self.pool(x, batch)
        out = self.head(g).view(-1)
        return out




def build_model(
    model_name: Literal["gcn", "sage", "gat", "gin"],
    cfg: ModelConfig,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "gcn":
        return GCNRegressor(cfg)
    if model_name == "sage":
        return GraphSAGERegressor(cfg)
    if model_name == "gat":
        return GATRegressor(cfg)
    if model_name == "gin":
        return GINRegressor(cfg)
    raise ValueError(f"Unknown model_name={model_name}")
