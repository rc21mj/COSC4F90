import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


REQUIRED_COLUMNS = [
    "timestamp",
    "vehicleId",
    "masterId",
    "candidateMasterId",
    "masterDistance",
    "candidateDistance",
    "masterRSSI",
    "candidateRSSI",
    "masterSINR",
    "candidateSINR",
    "masterRSRP",
    "candidateRSRP",
    "masterSpeed",
    "candidateSpeed",
    "vehicleDirection",
    "vehiclePosition-x",
    "vehiclePosition-y",
    "towerload",
    "selectedTower",
]


def safe_get(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return float(default)


def normalize_angle_deg(angle: float) -> float:
    return math.sin(math.radians(angle)), math.cos(math.radians(angle))


@dataclass
class Standardizer:
    mean: Dict[str, float]
    std: Dict[str, float]

    def transform(self, col: str, value: float) -> float:
        m = self.mean.get(col, 0.0)
        s = self.std.get(col, 1.0)
        if s == 0:
            return value - m
        return (value - m) / s


def fit_standardizer(df: pd.DataFrame, cols: List[str]) -> Standardizer:
    mean = {}
    std = {}
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        mean[c] = float(vals.mean())
        std[c] = float(vals.std()) if float(vals.std()) > 1e-8 else 1.0
    return Standardizer(mean=mean, std=std)


class HandoverGraphSequenceDataset(Dataset):
    """
    Each sample:
      - a sequence of T graph snapshots for one vehicle
      - target = index of true next selected tower in the final candidate list

    Local graph at each timestep:
      node 0 = UE
      node 1 = serving/master tower
      node 2 = candidate tower
      optional extra towers can be added later

    For now, this version uses the two tower entities explicitly present in the row:
      - masterId
      - candidateMasterId
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 10,
        pred_horizon: int = 1,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        standardizer: Standardizer = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.split = split

        self.df = pd.read_csv(csv_path)

        # Normalize messy CSV headers:
        # Example: "vehiclePosition- x" -> "vehiclePosition-x"
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace("- ", "-", regex=False)
            .str.replace(" -", "-", regex=False)
        )

        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = self.df.sort_values(["vehicleId", "timestamp"]).reset_index(drop=True)

        self.feature_cols_to_scale = [
            "masterDistance",
            "candidateDistance",
            "masterRSSI",
            "candidateRSSI",
            "masterSINR",
            "candidateSINR",
            "masterRSRP",
            "candidateRSRP",
            "masterSpeed",
            "candidateSpeed",
            "vehiclePosition-x",
            "vehiclePosition-y",
            "towerload",
        ]

        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.split_df = self.df.iloc[:train_end].copy()
        elif split == "val":
            self.split_df = self.df.iloc[train_end:val_end].copy()
        elif split == "test":
            self.split_df = self.df.iloc[val_end:].copy()
        else:
            raise ValueError("split must be one of: train, val, test")

        if standardizer is None:
            train_df = self.df.iloc[:train_end].copy()
            self.standardizer = fit_standardizer(train_df, self.feature_cols_to_scale)
        else:
            self.standardizer = standardizer

        self.vehicle_groups = []
        for vid, grp in self.split_df.groupby("vehicleId"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            if len(grp) >= self.seq_len + self.pred_horizon:
                self.vehicle_groups.append((vid, grp))

        self.index_map = []
        for group_idx, (_, grp) in enumerate(self.vehicle_groups):
            max_start = len(grp) - self.seq_len - self.pred_horizon + 1
            for start in range(max_start):
                self.index_map.append((group_idx, start))

    def __len__(self):
        return len(self.index_map)

    def _build_snapshot(self, row: pd.Series) -> Data:
        # UE features
        ue_speed = self.standardizer.transform("masterSpeed", safe_get(row, "masterSpeed"))
        ue_x = self.standardizer.transform("vehiclePosition-x", safe_get(row, "vehiclePosition-x"))
        ue_y = self.standardizer.transform("vehiclePosition-y", safe_get(row, "vehiclePosition-y"))
        dir_sin, dir_cos = normalize_angle_deg(safe_get(row, "vehicleDirection"))

        # serving tower features
        m_dist = self.standardizer.transform("masterDistance", safe_get(row, "masterDistance"))
        m_rssi = self.standardizer.transform("masterRSSI", safe_get(row, "masterRSSI"))
        m_sinr = self.standardizer.transform("masterSINR", safe_get(row, "masterSINR"))
        m_rsrp = self.standardizer.transform("masterRSRP", safe_get(row, "masterRSRP"))
        m_load = self.standardizer.transform("towerload", safe_get(row, "towerload"))

        # candidate tower features
        c_dist = self.standardizer.transform("candidateDistance", safe_get(row, "candidateDistance"))
        c_rssi = self.standardizer.transform("candidateRSSI", safe_get(row, "candidateRSSI"))
        c_sinr = self.standardizer.transform("candidateSINR", safe_get(row, "candidateSINR"))
        c_rsrp = self.standardizer.transform("candidateRSRP", safe_get(row, "candidateRSRP"))
        c_load = self.standardizer.transform("towerload", safe_get(row, "towerload"))

        master_id = int(safe_get(row, "masterId", 0))
        cand_id = int(safe_get(row, "candidateMasterId", 0))

        # Node layout:
        # 0 = UE
        # 1 = serving tower
        # 2 = candidate tower
        x = torch.tensor(
            [
                # UE node
                [
                    ue_speed, dir_sin, dir_cos, ue_x, ue_y,
                    0.0, 0.0, 0.0, 0.0, 0.0,   # tower-only padded slots
                    1.0, 0.0,                  # is_ue, is_tower
                ],
                # Master tower node
                [
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    m_dist, m_rssi, m_sinr, m_rsrp, m_load,
                    0.0, 1.0,
                ],
                # Candidate tower node
                [
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    c_dist, c_rssi, c_sinr, c_rsrp, c_load,
                    0.0, 1.0,
                ],
            ],
            dtype=torch.float,
        )

        # Edges: UE <-> master, UE <-> candidate, master <-> candidate
        # edge_attr: [distance, delta_rsrp, delta_sinr, is_serving, is_candidate]
        edge_index = torch.tensor(
            [
                [0, 1, 0, 2, 1, 2, 2, 1],
                [1, 0, 2, 0, 2, 1, 1, 2],
            ],
            dtype=torch.long,
        )

        edge_attr = torch.tensor(
            [
                [m_dist, 0.0, 0.0, 1.0, 0.0],                      # UE -> master
                [m_dist, 0.0, 0.0, 1.0, 0.0],                      # master -> UE
                [c_dist, c_rsrp - m_rsrp, c_sinr - m_sinr, 0.0, 1.0],  # UE -> cand
                [c_dist, c_rsrp - m_rsrp, c_sinr - m_sinr, 0.0, 1.0],  # cand -> UE
                [abs(c_dist - m_dist), c_rsrp - m_rsrp, c_sinr - m_sinr, 0.0, 0.0],
                [abs(c_dist - m_dist), m_rsrp - c_rsrp, m_sinr - c_sinr, 0.0, 0.0],
                [abs(c_dist - m_dist), m_rsrp - c_rsrp, m_sinr - c_sinr, 0.0, 0.0],
                [abs(c_dist - m_dist), c_rsrp - m_rsrp, c_sinr - m_sinr, 0.0, 0.0],
            ],
            dtype=torch.float,
        )

        tower_ids = torch.tensor([master_id, cand_id], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.ue_index = torch.tensor(0, dtype=torch.long)
        data.tower_indices = torch.tensor([1, 2], dtype=torch.long)
        data.tower_ids = tower_ids
        return data

    def __getitem__(self, idx: int) -> Dict[str, object]:
        group_idx, start = self.index_map[idx]
        _, grp = self.vehicle_groups[group_idx]

        seq_rows = grp.iloc[start : start + self.seq_len]
        target_row = grp.iloc[start + self.seq_len + self.pred_horizon - 1]

        graph_sequence = [self._build_snapshot(r) for _, r in seq_rows.iterrows()]

        # Candidate set from final snapshot
        last_snapshot = graph_sequence[-1]
        candidate_tower_ids = last_snapshot.tower_ids.tolist()

        selected_tower = int(safe_get(target_row, "selectedTower", candidate_tower_ids[0]))

        if selected_tower not in candidate_tower_ids:
            # fallback: if selectedTower is not in [masterId, candidateMasterId],
            # use master if it matches, otherwise choose candidate if that matches,
            # otherwise default to current master.
            target_index = 0
        else:
            target_index = candidate_tower_ids.index(selected_tower)

        return {
            "graph_sequence": graph_sequence,
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "candidate_tower_ids": torch.tensor(candidate_tower_ids, dtype=torch.long),
            "selected_tower": torch.tensor(selected_tower, dtype=torch.long),
        }