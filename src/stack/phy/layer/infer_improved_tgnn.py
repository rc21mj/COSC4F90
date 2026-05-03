"""
infer_improved_tgnn.py
======================
Two modes:
  1. OFFLINE (default): runs on simulator_data.csv test split for evaluation
  2. LIVE (--live):     runs on runtime_tgnn_window.csv written by C++ simulator

The C++ calls this script every 15 sim ticks via runProperTGNN().
It writes runtime_tgnn_window.csv then calls:
    python3 infer_improved_tgnn.py --live --ckpt improved_tgnn_ckpt

Output files (both modes):
  outputTGNN_proper.txt  -- tower_id,score per candidate (read by readProperTGNNOutput)
  outputTGNN.txt         -- best gap scalar (legacy compatibility)
"""

import argparse
import json
import os

import torch
import pandas as pd

from graph_dataset import HandoverGraphSequenceDataset, Standardizer, safe_get, normalize_angle_deg
from train_improved_tgnn import ImprovedTGNN
from torch_geometric.data import Data


def load_standardizer(ckpt_dir):
    with open(os.path.join(ckpt_dir, "standardizer.json")) as f:
        p = json.load(f)
    return Standardizer(mean=p["mean"], std=p["std"])


def build_snapshot_from_row(row, std):
    """Build a graph snapshot from a single pandas row using the standardizer."""
    ue_speed = std.transform("masterSpeed",       safe_get(row, "masterSpeed"))
    ue_x     = std.transform("vehiclePosition-x", safe_get(row, "vehiclePosition-x"))
    ue_y     = std.transform("vehiclePosition-y", safe_get(row, "vehiclePosition-y"))
    dir_sin, dir_cos = normalize_angle_deg(safe_get(row, "vehicleDirection"))
    m_dist = std.transform("masterDistance",    safe_get(row, "masterDistance"))
    m_rssi = std.transform("masterRSSI",        safe_get(row, "masterRSSI"))
    m_sinr = std.transform("masterSINR",        safe_get(row, "masterSINR"))
    m_rsrp = std.transform("masterRSRP",        safe_get(row, "masterRSRP"))
    m_load = std.transform("towerload",         safe_get(row, "towerload"))
    c_dist = std.transform("candidateDistance", safe_get(row, "candidateDistance"))
    c_rssi = std.transform("candidateRSSI",     safe_get(row, "candidateRSSI"))
    c_sinr = std.transform("candidateSINR",     safe_get(row, "candidateSINR"))
    c_rsrp = std.transform("candidateRSRP",     safe_get(row, "candidateRSRP"))
    c_load = std.transform("towerload",         safe_get(row, "towerload"))
    master_id = int(safe_get(row, "masterId", 0))
    cand_id   = int(safe_get(row, "candidateMasterId", 0))

    x = torch.tensor([
        [ue_speed, dir_sin, dir_cos, ue_x, ue_y, 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., m_dist, m_rssi, m_sinr, m_rsrp, m_load, 0., 1.],
        [0., 0., 0., 0., 0., c_dist, c_rssi, c_sinr, c_rsrp, c_load, 0., 1.],
    ], dtype=torch.float)

    edge_index = torch.tensor([[0,1,0,2,1,2,2,1],[1,0,2,0,2,1,1,2]], dtype=torch.long)
    edge_attr  = torch.tensor([
        [m_dist, 0., 0., 1., 0.],
        [m_dist, 0., 0., 1., 0.],
        [c_dist, c_rsrp-m_rsrp, c_sinr-m_sinr, 0., 1.],
        [c_dist, c_rsrp-m_rsrp, c_sinr-m_sinr, 0., 1.],
        [abs(c_dist-m_dist), c_rsrp-m_rsrp, c_sinr-m_sinr, 0., 0.],
        [abs(c_dist-m_dist), m_rsrp-c_rsrp, m_sinr-c_sinr, 0., 0.],
        [abs(c_dist-m_dist), m_rsrp-c_rsrp, m_sinr-c_sinr, 0., 0.],
        [abs(c_dist-m_dist), c_rsrp-m_rsrp, c_sinr-m_sinr, 0., 0.],
    ], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.ue_index      = torch.tensor(0, dtype=torch.long)
    data.tower_indices = torch.tensor([1, 2], dtype=torch.long)
    data.tower_ids     = torch.tensor([master_id, cand_id], dtype=torch.long)
    return data


def run_live(args, model, std, device):
    """
    LIVE mode: read runtime_tgnn_window.csv written by C++.
    The window has seq_len rows for one vehicle with potentially multiple
    candidate towers. We group by candidateMasterId and score each candidate.
    """
    df = pd.read_csv(args.window)
    df.columns = (df.columns.str.strip()
                  .str.replace(r"\s+", " ", regex=True)
                  .str.replace("- ", "-", regex=False)
                  .str.replace(" -", "-", regex=False))

    # Drop same-tower rows and rows with sentinel selectedTower=-1
    df = df[df["masterId"] != df["candidateMasterId"]].copy()

    if df.empty:
        # No real candidates — write empty outputs and exit
        open(args.out, "w").close()
        open(args.scalar_out, "w").write("0.000000\n")
        print("[ImprovedTGNN] No real candidates in window. Staying on master.")
        return

    # Get the master tower and vehicle from the last row
    master_id = int(df["masterId"].iloc[-1])
    vehicle_id = int(df["vehicleId"].iloc[-1])

    results = []  # (candidate_id, score)

    # Score each unique candidate tower
    for cand_id, cand_rows in df.groupby("candidateMasterId"):
        cand_rows = cand_rows.sort_values("timestamp").reset_index(drop=True)

        # Use last seq_len rows (or pad with first row if not enough)
        seq_len = args.seq
        if len(cand_rows) < seq_len:
            pad = seq_len - len(cand_rows)
            cand_rows = pd.concat([cand_rows.iloc[[0]] * pad, cand_rows]).reset_index(drop=True)

        seq_rows = cand_rows.iloc[-seq_len:]
        graph_sequence = [build_snapshot_from_row(r, std) for _, r in seq_rows.iterrows()]

        for g in graph_sequence:
            g.x             = g.x.to(device)
            g.edge_index    = g.edge_index.to(device)
            g.edge_attr     = g.edge_attr.to(device)
            g.ue_index      = g.ue_index.to(device)
            g.tower_indices = g.tower_indices.to(device)

        with torch.no_grad():
            scores = model(graph_sequence)  # [2]: [master_score, cand_score]
            cand_score = float(scores[1].item())
            master_score = float(scores[0].item())

        results.append((cand_id, cand_score, master_score))

    # Write outputTGNN_proper.txt: master first, then all candidates
    with open(args.out, "w") as f_proper, open(args.scalar_out, "w") as f_scalar:
        # Master row (always index 0)
        if results:
            avg_master_score = sum(r[2] for r in results) / len(results)
        else:
            avg_master_score = 0.0
        f_proper.write(f"{master_id},{avg_master_score:.6f}\n")

        best_gap = -999.0
        for cand_id, cand_score, master_score in results:
            f_proper.write(f"{int(cand_id)},{cand_score:.6f}\n")
            gap = cand_score - master_score
            if gap > best_gap:
                best_gap = gap

        # Scalar output: best gap across all candidates
        f_scalar.write(f"{best_gap:.6f}\n")

    print(f"[ImprovedTGNN] Vehicle={vehicle_id} Master={master_id} "
          f"Candidates={[r[0] for r in results]} BestGap={best_gap:.3f}")
    print(f"[ImprovedTGNN] Written -> {args.out}  {args.scalar_out}")


def run_offline(args, model, std, device):
    """OFFLINE mode: run on simulator_data.csv test split for evaluation."""
    ds = HandoverGraphSequenceDataset(
        csv_path=args.csv, seq_len=args.seq, pred_horizon=1,
        split=args.split, standardizer=std,
    )
    print(f"[INFO] Offline eval on split={args.split}, {len(ds)} samples")

    correct = ho_count = ho_correct = ho_total = 0

    with open(args.out, "w") as f_proper, \
         open(args.scalar_out, "w") as f_scalar, \
         torch.no_grad():

        for i, sample in enumerate(ds):
            graph_sequence = sample["graph_sequence"]
            candidate_ids  = sample["candidate_tower_ids"]
            target_index   = int(sample["target_index"].item())

            for g in graph_sequence:
                g.x             = g.x.to(device)
                g.edge_index    = g.edge_index.to(device)
                g.edge_attr     = g.edge_attr.to(device)
                g.ue_index      = g.ue_index.to(device)
                g.tower_indices = g.tower_indices.to(device)

            scores   = model(graph_sequence)
            pred_idx = int(torch.argmax(scores).item())

            if i == 0:
                print(f"Sample 0: towers={candidate_ids.tolist()} "
                      f"scores={[f'{float(s):.3f}' for s in scores.cpu()]} "
                      f"pred={int(candidate_ids[pred_idx].item())} "
                      f"gold={int(candidate_ids[target_index].item())}")

            for tower_id, score in zip(candidate_ids.tolist(), scores.cpu().tolist()):
                f_proper.write(f"{int(tower_id)},{float(score):.6f}\n")

            master_score    = float(scores[0].item())
            best_cand_score = float(scores[1:].max().item())
            f_scalar.write(f"{best_cand_score - master_score:.6f}\n")

            correct  += int(pred_idx == target_index)
            ho_count += int(pred_idx != 0)
            if target_index != 0:
                ho_total   += 1
                ho_correct += int(pred_idx == target_index)

    total = len(ds)
    print(f"\n[RESULT] Accuracy  : {correct}/{total} = {correct/total:.4f}")
    print(f"[RESULT] HO rate   : {ho_count}/{total} = {ho_count/total*100:.1f}%")
    print(f"[RESULT] HO recall : {ho_correct}/{ho_total} = {ho_correct/max(ho_total,1)*100:.1f}%")
    print(f"[INFO]   -> {args.out}  {args.scalar_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default="simulator_data.csv")
    parser.add_argument("--window",      default="runtime_tgnn_window.csv",
                        help="Runtime window file written by C++ (live mode)")
    parser.add_argument("--ckpt",        default="improved_tgnn_ckpt")
    parser.add_argument("--seq",         type=int, default=5)
    parser.add_argument("--hidden",      type=int, default=96)
    parser.add_argument("--heads",       type=int, default=4)
    parser.add_argument("--out",         default="outputTGNN_proper.txt")
    parser.add_argument("--scalar-out",  default="outputTGNN.txt")
    parser.add_argument("--split",       default="test")
    parser.add_argument("--live",        action="store_true",
                        help="Live mode: read runtime_tgnn_window.csv from C++")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    std    = load_standardizer(args.ckpt)

    model = ImprovedTGNN(node_dim=12, edge_dim=5, hidden_dim=args.hidden, heads=args.heads).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.ckpt, "best_model.pt"), map_location=device))
    model.eval()

    if args.live:
        run_live(args, model, std, device)
    else:
        run_offline(args, model, std, device)


if __name__ == "__main__":
    main()
