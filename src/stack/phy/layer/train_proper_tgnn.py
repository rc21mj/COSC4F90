import os
import json
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATv2Conv

from graph_dataset import HandoverGraphSequenceDataset


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
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.snapshot_encoder = SnapshotEncoder(node_dim, edge_dim, hidden_dim, heads=2)

        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.ue_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tower_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_sequence: List[object]) -> torch.Tensor:
        ue_embeddings = []
        tower_embeddings_over_time = []

        for g in graph_sequence:
            h = self.snapshot_encoder(g.x, g.edge_index, g.edge_attr)
            ue_embeddings.append(h[g.ue_index])
            tower_embeddings_over_time.append(h[g.tower_indices])

        ue_seq = torch.stack(ue_embeddings, dim=0).unsqueeze(0)   # [1, T, H]
        out, _ = self.temporal(ue_seq)
        ue_final = out[:, -1, :]                                  # [1, H]

        final_tower_embs = tower_embeddings_over_time[-1]         # [K, H]
        ue_expand = ue_final.expand(final_tower_embs.size(0), -1)

        pair_feat = torch.cat(
            [
                self.ue_proj(ue_expand),
                self.tower_proj(final_tower_embs),
                self.ue_proj(ue_expand) - self.tower_proj(final_tower_embs),
            ],
            dim=-1,
        )

        scores = self.scorer(pair_feat).squeeze(-1)               # [K]
        return scores


def run_epoch(model, dataset, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for sample in dataset:
        graph_sequence = sample["graph_sequence"]
        target_index = sample["target_index"].to(device)

        for g in graph_sequence:
            g.x = g.x.to(device)
            g.edge_index = g.edge_index.to(device)
            g.edge_attr = g.edge_attr.to(device)
            g.ue_index = g.ue_index.to(device)
            g.tower_indices = g.tower_indices.to(device)

        if training:
            optimizer.zero_grad()

        scores = model(graph_sequence)
        loss = F.cross_entropy(scores.unsqueeze(0), target_index.unsqueeze(0))

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss.item())
        pred = int(torch.argmax(scores).item())
        gold = int(target_index.item())
        correct += int(pred == gold)
        total += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    csv_path = "simulator_data.csv"
    seq_len = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = HandoverGraphSequenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        pred_horizon=1,
        split="train",
    )
    val_ds = HandoverGraphSequenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        pred_horizon=1,
        split="val",
        standardizer=train_ds.standardizer,
    )
    test_ds = HandoverGraphSequenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        pred_horizon=1,
        split="test",
        standardizer=train_ds.standardizer,
    )

    model = ProperTGNN(node_dim=12, edge_dim=5, hidden_dim=64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_acc = -1.0
    save_dir = "proper_tgnn_ckpt"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, 31):
        train_loss, train_acc = run_epoch(model, train_ds, optimizer=optimizer, device=device)
        val_loss, val_acc = run_epoch(model, val_ds, optimizer=None, device=device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

            scaler_payload = {
                "mean": train_ds.standardizer.mean,
                "std": train_ds.standardizer.std,
            }
            with open(os.path.join(save_dir, "standardizer.json"), "w") as f:
                json.dump(scaler_payload, f, indent=2)

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device))
    test_loss, test_acc = run_epoch(model, test_ds, optimizer=None, device=device)
    print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()