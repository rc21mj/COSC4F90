import os
import json
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATv2Conv

from graph_dataset import HandoverGraphSequenceDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SnapshotEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, heads: int = 2):
        super().__init__()
        self.gnn1 = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
        )
        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        h1 = self.gnn1(x, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h2 = self.gnn2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        return h2


class ProperTGNN(nn.Module):
    """
    Temporal Graph Neural Network for handover tower selection.

    Architecture:
      1. GATv2 snapshot encoder  — produces per-node embeddings at each timestep
      2. UE GRU                  — captures UE trajectory over the sequence
      3. Tower GRU (FIX)         — captures each tower's signal trend over time
      4. Scoring head            — ranks candidate towers using both temporal streams
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.snapshot_encoder = SnapshotEncoder(node_dim, edge_dim, hidden_dim, heads=2)

        # GRU for the UE node trajectory
        self.ue_temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # ------------------------------------------------------------------ #
        # FIX: separate GRU for each tower's signal trend over time.          #
        # We share weights across towers — each tower's embedding sequence    #
        # is treated as one batch item.                                        #
        # ------------------------------------------------------------------ #
        self.tower_temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.ue_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.tower_proj = nn.Linear(hidden_dim, hidden_dim)

        # Scoring head: UE_temporal + tower_temporal + elementwise difference
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_sequence: List) -> torch.Tensor:
        """
        graph_sequence: list of T PyG Data objects, each with
            .x, .edge_index, .edge_attr, .ue_index, .tower_indices

        Returns: scores tensor of shape [K] (K = number of candidate towers)
        """
        ue_embeddings_over_time    = []   # list of T tensors, each [H]
        tower_embeddings_over_time = []   # list of T tensors, each [K, H]

        for g in graph_sequence:
            h = self.snapshot_encoder(g.x, g.edge_index, g.edge_attr)
            ue_embeddings_over_time.append(h[g.ue_index])          # [H]
            tower_embeddings_over_time.append(h[g.tower_indices])  # [K, H]

        T = len(graph_sequence)
        K = tower_embeddings_over_time[0].size(0)

        # ── UE temporal stream ──────────────────────────────────────────── #
        ue_seq = torch.stack(ue_embeddings_over_time, dim=0).unsqueeze(0)  # [1, T, H]
        ue_out, _ = self.ue_temporal(ue_seq)
        ue_final  = ue_out[:, -1, :]                                       # [1, H]

        # ── Tower temporal stream ────────────────────────────────────────── #
        # Stack tower embeddings: shape [K, T, H] then run GRU over T dim
        tower_seq = torch.stack(tower_embeddings_over_time, dim=1)         # [K, T, H]
        tower_out, _ = self.tower_temporal(tower_seq)                      # [K, T, H]
        tower_final  = tower_out[:, -1, :]                                 # [K, H]

        # ── Scoring head ─────────────────────────────────────────────────── #
        ue_expand = ue_final.expand(K, -1)                                 # [K, H]

        pair_feat = torch.cat(
            [
                self.ue_proj(ue_expand),
                self.tower_proj(tower_final),
                self.ue_proj(ue_expand) - self.tower_proj(tower_final),
            ],
            dim=-1,
        )                                                                   # [K, 3H]

        scores = self.scorer(pair_feat).squeeze(-1)                        # [K]
        return scores


def compute_class_weights(dataset) -> torch.Tensor:
    """
    Count label frequency over the whole dataset and return inverse-frequency
    weights for use in F.cross_entropy. Warns if heavily imbalanced.
    """
    counts = {}
    for sample in dataset:
        label = int(sample["target_index"].item())
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return None

    num_classes = max(counts.keys()) + 1
    total       = sum(counts.values())
    weights     = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))

    w = torch.tensor(weights, dtype=torch.float)

    # Warn if dominant class accounts for >70% of labels
    dominant_frac = max(counts.values()) / total
    if dominant_frac > 0.70:
        logger.warning(
            "Class imbalance detected: label %d accounts for %.1f%% of samples. "
            "Applying inverse-frequency class weights.",
            max(counts, key=counts.get),
            dominant_frac * 100,
        )

    logger.info("Label distribution: %s", counts)
    logger.info("Class weights: %s", {i: round(float(w[i]), 3) for i in range(num_classes)})
    return w


def run_epoch(model, dataset, optimizer=None, device="cpu", class_weights=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    if class_weights is not None:
        class_weights = class_weights.to(device)

    total_loss = 0.0
    correct    = 0
    total      = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for sample in dataset:
            graph_sequence = sample["graph_sequence"]
            target_index   = sample["target_index"].to(device)

            for g in graph_sequence:
                g.x           = g.x.to(device)
                g.edge_index  = g.edge_index.to(device)
                g.edge_attr   = g.edge_attr.to(device)
                g.ue_index    = g.ue_index.to(device)
                g.tower_indices = g.tower_indices.to(device)

            if training:
                optimizer.zero_grad()

            scores = model(graph_sequence)
            loss   = F.cross_entropy(
                scores.unsqueeze(0),
                target_index.unsqueeze(0),
                weight=class_weights,
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item())
            pred    = int(torch.argmax(scores).item())
            gold    = int(target_index.item())
            correct += int(pred == gold)
            total   += 1

    avg_loss = total_loss / max(total, 1)
    acc      = correct    / max(total, 1)
    return avg_loss, acc


def main():
    csv_path   = "simulator_data.csv"
    seq_len    = 10
    max_epochs = 100       # was 30 — give the model room to converge
    patience   = 15        # early stopping patience
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir   = "proper_tgnn_ckpt"
    os.makedirs(save_dir, exist_ok=True)

    train_ds = HandoverGraphSequenceDataset(
        csv_path=csv_path, seq_len=seq_len, pred_horizon=1, split="train",
    )
    val_ds = HandoverGraphSequenceDataset(
        csv_path=csv_path, seq_len=seq_len, pred_horizon=1, split="val",
        standardizer=train_ds.standardizer,
    )
    test_ds = HandoverGraphSequenceDataset(
        csv_path=csv_path, seq_len=seq_len, pred_horizon=1, split="test",
        standardizer=train_ds.standardizer,
    )

    logger.info("Train samples: %d | Val: %d | Test: %d",
                len(train_ds), len(val_ds), len(test_ds))

    # Compute class weights on training set
    class_weights = compute_class_weights(train_ds)

    model     = ProperTGNN(node_dim=12, edge_dim=5, hidden_dim=64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # LR scheduler: halve LR when val_loss stops improving
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_acc  = -1.0
    epochs_no_imp = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_ds, optimizer=optimizer,
            device=device, class_weights=class_weights,
        )
        val_loss, val_acc = run_epoch(
            model, val_ds, optimizer=None, device=device,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f | lr=%.2e",
            epoch, train_loss, train_acc, val_loss, val_acc, current_lr,
        )

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            epochs_no_imp = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            scaler_payload = {
                "mean": train_ds.standardizer.mean,
                "std":  train_ds.standardizer.std,
            }
            with open(os.path.join(save_dir, "standardizer.json"), "w") as f:
                json.dump(scaler_payload, f, indent=2)
            logger.info("  -> New best val_acc=%.4f, model saved.", best_val_acc)
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs).",
                            epoch, patience)
                break

    # Final test evaluation
    model.load_state_dict(
        torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    )
    test_loss, test_acc = run_epoch(model, test_ds, optimizer=None, device=device)
    logger.info("TEST | loss=%.4f acc=%.4f", test_loss, test_acc)

    # Report fallback rates (how often selectedTower was outside candidate set)
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        _ = [ds[i] for i in range(len(ds))]  # ensure all items accessed
    logger.info(
        "Fallback rates — train: %.2f%% | val: %.2f%% | test: %.2f%%",
        train_ds.fallback_rate() * 100,
        val_ds.fallback_rate()   * 100,
        test_ds.fallback_rate()  * 100,
    )


if __name__ == "__main__":
    main()
