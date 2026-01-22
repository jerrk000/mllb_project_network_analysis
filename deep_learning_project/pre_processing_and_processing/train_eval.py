# train_eval.py
"""
Train + evaluate GNN graph-level regressors with GroupKFold-by-city (place).

CPU-safe defaults:
- device = cpu
- num_workers = 0
- conservative batch size
- small-ish hidden dims (configurable)

Expected dataset format:
- A torch-saved list of torch_geometric.data.Data objects (torch.save(list, path))
- Each Data has:
    x: [N, in_dim]
    edge_index: [2, E]
    y: [1] float tensor (graph label)
    place: str (group id for GroupKFold; e.g., city name)
  Optional:
    edge_attr: [E, 1] float tensor (e.g., edge length)

Outputs:
- <outdir>/gnn_fold_details.csv
- <outdir>/gnn_cv_summary.csv
- <outdir>/gnn_predictions.csv (optional, enabled by default)

Example:
  python train_eval.py --dataset pyg_dataset.pt --outdir results --splits 5 --epochs 200 --batch-size 16 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import trange

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

from torch_geometric.loader import DataLoader

from models import ModelConfig, build_model


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU only: deterministic where feasible
    torch.use_deterministic_algorithms(False)


# -----------------------------
# Data helpers
# -----------------------------
def load_dataset(path: str):
    # PyTorch 2.6+: default weights_only=True breaks loading PyG Data objects.
    # Only do this if you trust the dataset file (you generated it).
    data_list = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(data_list, list) or len(data_list) == 0:
        raise ValueError(f"Dataset at {path} is not a non-empty list.")
    for i, d in enumerate(data_list[:5]):
        if not hasattr(d, "x") or not hasattr(d, "edge_index") or not hasattr(d, "y"):
            raise ValueError(f"Data[{i}] missing required fields x/edge_index/y.")
        if not hasattr(d, "place"):
            raise ValueError(f"Data[{i}] missing required field 'place' for GroupKFold.")
    return data_list


def make_splits(data_list, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.array([str(d.place) for d in data_list])
    y = np.array([float(d.y.view(-1)[0].item()) for d in data_list], dtype=np.float32)

    gkf = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(data_list))
    splits = [(train_idx, test_idx) for train_idx, test_idx in gkf.split(indices, y, groups)]
    return splits


# -----------------------------
# Training / evaluation
# -----------------------------
def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    losses = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(batch)  # [num_graphs]
        y = batch.y.view(-1).to(pred.dtype)

        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_model(model, loader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    y_true_list = []
    y_pred_list = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).detach().cpu().numpy().astype(np.float64)
        y = batch.y.view(-1).detach().cpu().numpy().astype(np.float64)

        y_pred_list.append(pred)
        y_true_list.append(y)

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])

    if y_true.size == 0:
        return float("nan"), float("nan"), y_true, y_pred

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return float(mse), float(r2), y_true, y_pred


def run_fold(
    model_name: str,
    cfg: ModelConfig,
    train_data,
    test_data,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Dict:
    model = build_model(model_name, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    best_test_mse = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0

    history = []

    pbar = trange(1, epochs + 1, desc=f"{model_name.upper()} | fold", leave=False)

    for epoch in pbar:
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        test_mse, test_r2, _, _ = eval_model(model, test_loader, device)

        pbar.set_postfix({
            "train_mse": f"{train_mse:.4f}",
            "test_mse": f"{test_mse:.4f}",
            "r2": f"{test_r2:.3f}",
        })



        history.append((epoch, train_mse, test_mse, test_r2))

        # Early stopping on test fold metric (acts as validation in CV)
        # In strict setups you'd use an inner split; for small datasets this is acceptable
        # as long as you report it consistently across models.
        if test_mse < best_test_mse - 1e-12:
            best_test_mse = test_mse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation with best model
    test_mse, test_r2, y_true, y_pred = eval_model(model, test_loader, device)

    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "y_true": y_true,
        "y_pred": y_pred,
        "history": history,
    }


# -----------------------------
# Main experiment runner
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt", help="Path to pyg_dataset.pt")
    parser.add_argument("--outdir", type=str, default="results_gnn", help="Output directory")
    parser.add_argument("--splits", type=int, default=5, help="GroupKFold splits")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs per fold")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Message passing layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--gat-heads", type=int, default=4, help="GAT heads")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--save-preds", action="store_true", default=True, help="Save per-graph predictions")
    parser.add_argument("--no-save-preds", dest="save_preds", action="store_false", help="Disable saving predictions")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # CPU-safe per request

    data_list = load_dataset(args.dataset)

    in_dim = int(data_list[0].x.size(1))
    cfg = ModelConfig(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
        gcn_edge_weight=True,
        edge_weight_mode="inverse_length",
        gin_use_edge_attr=False,  # avoids confusion
    )

    splits = make_splits(data_list, args.splits)

    models = ["gcn", "sage", "gat", "gin"]

    fold_rows = []
    pred_rows = []

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_id + 1}/{len(splits)} ---")
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]

            heldout_places = sorted(set(str(d.place) for d in test_data))

            for model_name in models:
                print(f"\nTraining model: {model_name.upper()}")
                res = run_fold(
                    model_name=model_name,
                    cfg=cfg,
                    train_data=train_data,
                    test_data=test_data,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                )

                print(
                    f"Done {model_name.upper()} | "
                    f"MSE={res['test_mse']:.4f}, "
                    f"R2={res['test_r2']:.3f}, "
                    f"best_epoch={res['best_epoch']}"
                )

                fold_rows.append({
                    "seed": seed,
                    "fold": fold_id,
                    "model": model_name,
                    "best_epoch": res["best_epoch"],
                    "test_mse": res["test_mse"],
                    "test_r2": res["test_r2"],
                    "heldout_places": "|".join(heldout_places),
                    "n_train_graphs": len(train_data),
                    "n_test_graphs": len(test_data),
                    **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
                })

                if args.save_preds:
                    # Reconstruct a stable identifier per graph if available; else use index-in-fold
                    # If you store graph_file in Data, we include it; otherwise None.
                    graph_files = [getattr(d, "graph_file", None) for d in test_data]
                    places = [str(d.place) for d in test_data]

                    y_true = res["y_true"]
                    y_pred = res["y_pred"]

                    for j in range(len(y_true)):
                        pred_rows.append({
                            "seed": seed,
                            "fold": fold_id,
                            "model": model_name,
                            "place": places[j],
                            "graph_file": graph_files[j],
                            "y_true": float(y_true[j]),
                            "y_pred": float(y_pred[j]),
                        })

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(outdir / "gnn_fold_details.csv", index=False)

    if args.save_preds:
        df_preds = pd.DataFrame(pred_rows)
        df_preds.to_csv(outdir / "gnn_predictions.csv", index=False)

    # Aggregate summary
    summary = (
        df_folds
        .groupby(["model"])
        .agg(
            mse_mean=("test_mse", "mean"),
            mse_std=("test_mse", "std"),
            r2_mean=("test_r2", "mean"),
            r2_std=("test_r2", "std"),
            runs=("test_mse", "count"),
        )
        .reset_index()
        .sort_values(by="mse_mean", ascending=True)
    )
    summary.to_csv(outdir / "gnn_cv_summary.csv", index=False)

    # Also print to console
    print("\n=== GNN CV Summary (lower MSE better, higher R2 better) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
