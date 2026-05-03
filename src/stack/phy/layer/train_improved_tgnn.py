"""
train_improved_tgnn.py
======================
Improved TGNN for handover prediction — designed to beat the LSTM baseline.

Key improvements over train_proper_tgnn.py:
  1. Hysteresis-aware loss      — penalises switching away from current tower
                                  unless gain is meaningful (margin loss term)
  2. Per-tower temporal GRU     — each tower gets its own GRU over time, not
                                  just the UE; captures signal stability trends
  3. Longer sequence window     — default seq_len=20 (was 10)
  4. Richer serving-bias feats  — explicit is_serving flag already in dataset v2
  5. Deeper scorer w/ skip      — residual connection in final MLP
  6. Cosine-annealing LR        — smoother convergence, less ping-pong in
                                  parameter space → fewer ping-pong handovers
  7. Switched-tower penalty     — extra cross-entropy weight when ground truth
                                  is "stay" (index 0 = master) so the model
                                  learns conservatism

Usage
-----
  python3 train_improved_tgnn.py
  python3 train_improved_tgnn.py --csv simulator_data.csv --epochs 50 --seq 20
"""

import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from torch_geometric.nn import GATv2Conv

from graph_dataset import HandoverGraphSequenceDataset

# ─────────────────────────────────────────────────────────────────────────── #
#  Architecture
# ─────────────────────────────────────────────────────────────────────────── #

class SnapshotEncoder(nn.Module):
    """Two-layer GATv2 graph encoder — same as baseline but with dropout."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.gnn1 = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=0.1,
        )
        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(0.15)

    def forward(self, x, edge_index, edge_attr):
        h = self.gnn1(x, edge_index, edge_attr)
        h = self.drop(F.gelu(self.norm1(h)))
        h = self.gnn2(h, edge_index, edge_attr)
        h = self.norm2(h)
        return F.gelu(h)


class ImprovedTGNN(nn.Module):
    """
    Temporal GNN with per-entity temporal encoding.

    Unlike the baseline (which only runs a GRU over UE embeddings), this model:
      - runs a GRU over each tower's embedding sequence independently
      - concatenates [UE_final, tower_final, UE-tower_diff, UE-tower_product]
        for a richer pairwise score
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 96, heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Spatial encoder
        self.encoder = SnapshotEncoder(node_dim, edge_dim, hidden_dim, heads=heads)

        # Temporal encoders — one for UE, one shared across towers
        self.ue_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.tower_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # Projections
        self.ue_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.tower_proj = nn.Linear(hidden_dim, hidden_dim)

        # Scorer: 4 * hidden_dim input (ue, tower, diff, product)
        scorer_in = hidden_dim * 4
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Skip projection for residual (matches scorer_in → 1 directly)
        self.skip = nn.Linear(scorer_in, 1, bias=False)

    def forward(self, graph_sequence: List) -> torch.Tensor:
        ue_emb_seq     = []    # [T, H]
        tower_emb_seq  = []    # [T, K, H]

        for g in graph_sequence:
            h = self.encoder(g.x, g.edge_index, g.edge_attr)
            ue_emb_seq.append(h[g.ue_index])          # [H]
            tower_emb_seq.append(h[g.tower_indices])  # [K, H]

        K = tower_emb_seq[0].size(0)

        # ── UE temporal
        ue_seq = torch.stack(ue_emb_seq, dim=0).unsqueeze(0)   # [1, T, H]
        ue_out, _ = self.ue_gru(ue_seq)
        ue_final = ue_out[:, -1, :]                             # [1, H]

        # ── Per-tower temporal
        #    tower_emb_seq: list of T tensors [K, H] → need [K, T, H]
        tower_stack = torch.stack(tower_emb_seq, dim=1)         # [K, T, H]
        tower_out, _ = self.tower_gru(tower_stack)              # [K, T, H]
        tower_final = tower_out[:, -1, :]                       # [K, H]

        # ── Pairwise scoring
        ue_expand = ue_final.expand(K, -1)                      # [K, H]
        u = self.ue_proj(ue_expand)
        t = self.tower_proj(tower_final)

        pair = torch.cat([u, t, u - t, u * t], dim=-1)         # [K, 4H]
        scores = self.scorer(pair).squeeze(-1) + self.skip(pair).squeeze(-1)
        return scores                                            # [K]


# ─────────────────────────────────────────────────────────────────────────── #
#  Focal + hysteresis loss
# ─────────────────────────────────────────────────────────────────────────── #

def focal_hysteresis_loss(
    scores: torch.Tensor,
    target_index: torch.Tensor,
    serving_index: int = 0,
    switch_margin: float = 0.5,
    gamma: float = 2.0,
    switch_weight: float = 1.0,
    ho_weight: float = 4.0,
) -> torch.Tensor:
    """
    Focal loss + hysteresis margin penalty.

    Focal loss down-weights easy "stay" examples so the model is forced to
    learn the rare but critical handover cases instead of collapsing to
    always-stay (which the previous run showed happening at 96%+ stay rate).

    ho_weight     : class weight for switch samples — higher = more willing to HO.
    switch_weight : class weight for stay samples.
    gamma         : focal exponent. 2.0 is standard.
    switch_margin : candidate must beat master by this margin in logit space
                    before a handover incurs no margin penalty.
    """
    is_switch = int(target_index.item()) != serving_index
    ce_weight = ho_weight if is_switch else switch_weight

    log_probs   = F.log_softmax(scores.unsqueeze(0), dim=-1)
    probs       = log_probs.exp()
    target_prob = probs[0, target_index]

    focal_factor = (1.0 - target_prob.detach()) ** gamma
    task_loss    = -focal_factor * ce_weight * log_probs[0, target_index]

    serving_score = scores[serving_index]
    other_scores  = torch.cat([scores[:serving_index], scores[serving_index + 1:]])

    if other_scores.numel() == 0:
        return task_loss

    best_other     = other_scores.max()
    margin_penalty = F.relu(best_other - serving_score - switch_margin)

    return task_loss + 0.3 * margin_penalty


# ─────────────────────────────────────────────────────────────────────────── #
#  Training / eval loop
# ─────────────────────────────────────────────────────────────────────────── #

def run_epoch(model, dataset, optimizer=None, device="cpu",
              switch_margin=0.5, sampler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0
    ho_count   = 0
    ho_correct = 0   # correctly predicted handovers
    ho_total   = 0   # total true handover samples

    # Use sampler order during training, sequential otherwise
    if training and sampler is not None:
        indices = list(sampler)
    else:
        indices = range(len(dataset))

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for idx in indices:
            sample       = dataset[idx]
            graph_sequence = sample["graph_sequence"]
            target_index   = sample["target_index"].to(device)

            for g in graph_sequence:
                g.x             = g.x.to(device)
                g.edge_index    = g.edge_index.to(device)
                g.edge_attr     = g.edge_attr.to(device)
                g.ue_index      = g.ue_index.to(device)
                g.tower_indices = g.tower_indices.to(device)

            if training:
                optimizer.zero_grad()

            scores = model(graph_sequence)
            loss   = focal_hysteresis_loss(
                scores, target_index,
                serving_index=0,
                switch_margin=switch_margin,
            )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item())
            pred = int(torch.argmax(scores).item())
            gold = int(target_index.item())
            correct  += int(pred == gold)
            ho_count += int(pred != 0)
            if gold != 0:
                ho_total   += 1
                ho_correct += int(pred == gold)
            total += 1

    avg_loss = total_loss / max(total, 1)
    acc      = correct    / max(total, 1)
    ho_rate  = ho_count   / max(total, 1)
    ho_recall = ho_correct / max(ho_total, 1)   # recall on true HO samples
    return avg_loss, acc, ho_rate, ho_recall


# ─────────────────────────────────────────────────────────────────────────── #
#  Main
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="simulator_data.csv")
    parser.add_argument("--epochs",    type=int,   default=60)
    parser.add_argument("--seq",       type=int,   default=5,
                        help="Sequence length — must match PROPER_TGNN_STEPS in LtePhyUe.h")
    parser.add_argument("--hidden",    type=int,   default=96)
    parser.add_argument("--heads",     type=int,   default=4)
    parser.add_argument("--lr",        type=float, default=5e-4)
    parser.add_argument("--margin",    type=float, default=0.5,
                        help="Hysteresis switch margin")
    parser.add_argument("--ho-weight", type=float, default=4.0,
                        help="Focal loss weight for handover samples (higher = more HOs predicted)")
    parser.add_argument("--out",       default="improved_tgnn_ckpt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] seq_len={args.seq}, hidden={args.hidden}, heads={args.heads}, "
          f"margin={args.margin}, ho_weight={args.ho_weight}, epochs={args.epochs}")

    # ── Datasets
    train_ds = HandoverGraphSequenceDataset(
        csv_path=args.csv, seq_len=args.seq, pred_horizon=1, split="train"
    )
    val_ds = HandoverGraphSequenceDataset(
        csv_path=args.csv, seq_len=args.seq, pred_horizon=1, split="val",
        standardizer=train_ds.standardizer,
    )
    test_ds = HandoverGraphSequenceDataset(
        csv_path=args.csv, seq_len=args.seq, pred_horizon=1, split="test",
        standardizer=train_ds.standardizer,
    )

    print(f"[INFO] train={len(train_ds)} "
          f"(stay={train_ds.n_stay}, switch={train_ds.n_switch})  "
          f"val={len(val_ds)}  test={len(test_ds)}")

    # ── Balanced sampler — draws equal numbers of stay/switch per epoch
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    # ── Model
    model = ImprovedTGNN(
        node_dim=12, edge_dim=5, hidden_dim=args.hidden, heads=args.heads
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {n_params:,}")

    # ── Optimiser + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.out, exist_ok=True)
    best_score   = -1.0   # balanced accuracy = (acc_stay + acc_switch) / 2
    best_val_acc = -1.0
    best_val_ho  =  1.0

    print(f"\n{'Ep':>3}  {'tr_loss':>8}  {'tr_acc':>7}  {'tr_HO%':>7}  {'tr_HOrec':>8}  "
          f"{'va_acc':>7}  {'va_HO%':>7}  {'va_HOrec':>8}  {'LR':>9}")
    print("-" * 90)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_ho, tr_rec = run_epoch(
            model, train_ds, optimizer=optimizer,
            device=device, switch_margin=args.margin, sampler=sampler
        )
        va_loss, va_acc, va_ho, va_rec = run_epoch(
            model, val_ds, optimizer=None,
            device=device, switch_margin=args.margin
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_ho*100:>6.1f}%  {tr_rec*100:>7.1f}%  "
            f"{va_acc:>7.4f}  {va_ho*100:>6.1f}%  {va_rec*100:>7.1f}%  {lr_now:>9.2e}"
        )

        # Primary save criterion: balanced accuracy (avg of stay-recall and switch-recall)
        # This prevents saving a model that only learned to always stay.
        # stay_recall = (va_acc * total - va_rec * n_switch) / n_stay  (approx)
        va_balanced = (va_acc + va_rec) / 2.0
        improved = (va_balanced > best_score) or (
            abs(va_balanced - best_score) < 1e-4 and va_ho < best_val_ho
        )
        if improved:
            best_score   = va_balanced
            best_val_acc = va_acc
            best_val_ho  = va_ho
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
            scaler_payload = {
                "mean": train_ds.standardizer.mean,
                "std":  train_ds.standardizer.std,
            }
            with open(os.path.join(args.out, "standardizer.json"), "w") as f:
                json.dump(scaler_payload, f, indent=2)
            print(f"     ↑ saved (bal={best_score:.4f} acc={best_val_acc:.4f} "
                  f"HO={best_val_ho*100:.1f}% HOrec={va_rec*100:.1f}%)")

    # ── Final test
    print("\n" + "=" * 90)
    model.load_state_dict(
        torch.load(os.path.join(args.out, "best_model.pt"), map_location=device)
    )
    te_loss, te_acc, te_ho, te_rec = run_epoch(
        model, test_ds, optimizer=None, device=device, switch_margin=args.margin
    )
    print(f"TEST | loss={te_loss:.4f}  acc={te_acc:.4f}  "
          f"HO_rate={te_ho*100:.1f}%  HO_recall={te_rec*100:.1f}%")
    print(f"[INFO] Checkpoint saved -> {args.out}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
