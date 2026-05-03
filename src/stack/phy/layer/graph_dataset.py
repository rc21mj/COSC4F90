"""
graph_dataset.py — final version, faithful to PhD student's original logic
==========================================================================
DO NOT CHANGE the label/grouping logic. This matches the original exactly.
The only additions are:
  - sample_weights / n_stay / n_switch for WeightedRandomSampler support
  - minor code cleanup (no behaviour change)

Label (unchanged from original):
  selectedTower is in [masterId, candidateMasterId] for each row.
  target_index = candidate_tower_ids.index(selectedTower)
    → 0 if selectedTower == masterId  (stay)
    → 1 if selectedTower == candidateMasterId (switch)
  If selectedTower not in candidate list → default to 0 (stay).

Grouping (unchanged from original):
  Per vehicleId, sorted by timestamp. Same as PhD student's code.
"""

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


REQUIRED_COLUMNS = [
    "timestamp", "vehicleId", "masterId", "candidateMasterId",
    "masterDistance", "candidateDistance", "masterRSSI", "candidateRSSI",
    "masterSINR", "candidateSINR", "masterRSRP", "candidateRSRP",
    "masterSpeed", "candidateSpeed", "vehicleDirection",
    "vehiclePosition-x", "vehiclePosition-y", "towerload", "selectedTower",
]


def safe_get(row, col, default=0.0):
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return float(default)


def normalize_angle_deg(angle):
    return math.sin(math.radians(angle)), math.cos(math.radians(angle))


@dataclass
class Standardizer:
    mean: Dict[str, float]
    std: Dict[str, float]

    def transform(self, col, value):
        m = self.mean.get(col, 0.0)
        s = self.std.get(col, 1.0)
        return (value - m) / s if s != 0 else value - m


def fit_standardizer(df, cols):
    mean, std = {}, {}
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        mean[c] = float(vals.mean())
        std[c] = float(vals.std()) if float(vals.std()) > 1e-8 else 1.0
    return Standardizer(mean=mean, std=std)


class HandoverGraphSequenceDataset(Dataset):

    def __init__(self, csv_path, seq_len=5, pred_horizon=1, split="train",
                 train_ratio=0.7, val_ratio=0.15, standardizer=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        df = pd.read_csv(csv_path)
        df.columns = (df.columns.str.strip()
                      .str.replace(r"\s+", " ", regex=True)
                      .str.replace("- ", "-", regex=False)
                      .str.replace(" -", "-", regex=False))

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Original sort: vehicleId + timestamp (same as PhD student)
        df = df.sort_values(["vehicleId", "timestamp"]).reset_index(drop=True)

        self.feature_cols_to_scale = [
            "masterDistance", "candidateDistance",
            "masterRSSI", "candidateRSSI",
            "masterSINR", "candidateSINR",
            "masterRSRP", "candidateRSRP",
            "masterSpeed", "candidateSpeed",
            "vehiclePosition-x", "vehiclePosition-y", "towerload",
        ]

        n = len(df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.split_df = df.iloc[:train_end].copy()
        elif split == "val":
            self.split_df = df.iloc[train_end:val_end].copy()
        elif split == "test":
            self.split_df = df.iloc[val_end:].copy()
        else:
            raise ValueError("split must be train/val/test")

        if standardizer is None:
            self.standardizer = fit_standardizer(df.iloc[:train_end], self.feature_cols_to_scale)
        else:
            self.standardizer = standardizer

        # Per-vehicle groups (original logic)
        self.vehicle_groups = []
        for vid, grp in self.split_df.groupby("vehicleId"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            if len(grp) >= seq_len + pred_horizon:
                self.vehicle_groups.append((vid, grp))

        self.index_map = []
        for gi, (_, grp) in enumerate(self.vehicle_groups):
            for start in range(len(grp) - seq_len - pred_horizon + 1):
                self.index_map.append((gi, start))

        self._build_sample_weights()

        if split == "train":
            print(f"[INFO] train={len(self.index_map)} "
                  f"(stay={self.n_stay}, switch={self.n_switch})")

    def _get_label(self, grp, start):
        target_row = grp.iloc[start + self.seq_len + self.pred_horizon - 1]
        last_row   = grp.iloc[start + self.seq_len - 1]
        master_id  = int(safe_get(last_row, "masterId", 0))
        cand_id    = int(safe_get(last_row, "candidateMasterId", 0))
        selected   = int(safe_get(target_row, "selectedTower", master_id))
        candidate_tower_ids = [master_id, cand_id]
        if selected not in candidate_tower_ids:
            return 0
        return candidate_tower_ids.index(selected)

    def _build_sample_weights(self):
        labels = [self._get_label(*self._lookup(i)) for i in range(len(self.index_map))]
        labels = np.array(labels)
        self.n_stay   = int((labels == 0).sum())
        self.n_switch = int((labels == 1).sum())
        total = len(labels)
        if self.n_stay == 0 or self.n_switch == 0:
            self.sample_weights = torch.ones(total)
        else:
            w_stay   = total / (2.0 * self.n_stay)
            w_switch = total / (2.0 * self.n_switch)
            weights  = np.where(labels == 0, w_stay, w_switch)
            self.sample_weights = torch.tensor(weights, dtype=torch.float)

    def _lookup(self, idx):
        gi, start = self.index_map[idx]
        _, grp = self.vehicle_groups[gi]
        return grp, start

    def __len__(self):
        return len(self.index_map)

    def _build_snapshot(self, row):
        ue_speed = self.standardizer.transform("masterSpeed",       safe_get(row, "masterSpeed"))
        ue_x     = self.standardizer.transform("vehiclePosition-x", safe_get(row, "vehiclePosition-x"))
        ue_y     = self.standardizer.transform("vehiclePosition-y", safe_get(row, "vehiclePosition-y"))
        dir_sin, dir_cos = normalize_angle_deg(safe_get(row, "vehicleDirection"))
        m_dist = self.standardizer.transform("masterDistance",    safe_get(row, "masterDistance"))
        m_rssi = self.standardizer.transform("masterRSSI",        safe_get(row, "masterRSSI"))
        m_sinr = self.standardizer.transform("masterSINR",        safe_get(row, "masterSINR"))
        m_rsrp = self.standardizer.transform("masterRSRP",        safe_get(row, "masterRSRP"))
        m_load = self.standardizer.transform("towerload",         safe_get(row, "towerload"))
        c_dist = self.standardizer.transform("candidateDistance", safe_get(row, "candidateDistance"))
        c_rssi = self.standardizer.transform("candidateRSSI",     safe_get(row, "candidateRSSI"))
        c_sinr = self.standardizer.transform("candidateSINR",     safe_get(row, "candidateSINR"))
        c_rsrp = self.standardizer.transform("candidateRSRP",     safe_get(row, "candidateRSRP"))
        c_load = self.standardizer.transform("towerload",         safe_get(row, "towerload"))
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

    def __getitem__(self, idx):
        gi, start = self.index_map[idx]
        _, grp = self.vehicle_groups[gi]
        seq_rows   = grp.iloc[start : start + self.seq_len]
        target_row = grp.iloc[start + self.seq_len + self.pred_horizon - 1]

        graph_sequence = [self._build_snapshot(r) for _, r in seq_rows.iterrows()]
        last_snap      = graph_sequence[-1]
        candidate_tower_ids = last_snap.tower_ids.tolist()
        master_id  = candidate_tower_ids[0]
        selected   = int(safe_get(target_row, "selectedTower", master_id))
        if selected not in candidate_tower_ids:
            target_index = 0
        else:
            target_index = candidate_tower_ids.index(selected)

        return {
            "graph_sequence":      graph_sequence,
            "target_index":        torch.tensor(target_index, dtype=torch.long),
            "candidate_tower_ids": torch.tensor(candidate_tower_ids, dtype=torch.long),
            "selected_tower":      torch.tensor(selected, dtype=torch.long),
        }
