"""
graph_dataset.py
================
Handover graph sequence dataset for the ProperTGNN model.

Key design decisions that eliminate the 89% fallback rate:

1.  CSV structure:  Each (timestamp, vehicleId) pair appears ONCE per candidate
    tower, so many rows share the same timestamp but have different
    candidateMasterId.  We group by (vehicleId, timestamp) first to get the
    full set of candidate towers seen at each tick, THEN build sequences.

2.  selectedTower = 0:  Means "no handover this tick — stay on master."
    This is NOT a missing value.  It maps to label index 0 (master tower).

3.  Candidate set:  Built from ALL candidateMasterId values seen for a given
    (vehicleId, timestamp) tick, plus the masterId.  The label is the index
    of selectedTower in this full candidate set.

4.  Sequence construction:  seq_len consecutive ticks for one vehicle.
    The label comes from the tick immediately after the sequence ends.
    Each snapshot uses the BEST row for that tick (highest candidateRSSI)
    to represent the radio environment at that moment.

5.  Train/val/test split:  By vehicleId (not row index) to prevent leakage.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "timestamp", "vehicleId", "masterId", "candidateMasterId",
    "masterDistance", "candidateDistance",
    "masterRSSI", "candidateRSSI",
    "masterSINR", "candidateSINR",
    "masterRSRP", "candidateRSRP",
    "masterSpeed", "candidateSpeed",
    "vehicleDirection",
    "vehiclePosition-x", "vehiclePosition-y",
    "towerload", "selectedTower",
]


def safe_get(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col in row and pd.notna(row[col]):
        try:
            return float(row[col])
        except (ValueError, TypeError):
            return float(default)
    return float(default)


def normalize_angle_deg(angle: float) -> Tuple[float, float]:
    return math.sin(math.radians(angle)), math.cos(math.radians(angle))


@dataclass
class Standardizer:
    mean: Dict[str, float]
    std:  Dict[str, float]

    def transform(self, col: str, value: float) -> float:
        m = self.mean.get(col, 0.0)
        s = self.std.get(col, 1.0)
        if s == 0:
            return value - m
        return (value - m) / s


def fit_standardizer(df: pd.DataFrame, cols: List[str]) -> Standardizer:
    mean, std = {}, {}
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        mean[c] = float(vals.mean())
        s = float(vals.std())
        std[c]  = s if s > 1e-8 else 1.0
    return Standardizer(mean=mean, std=std)


# ─────────────────────────────────────────────────────────────────────────── #
# Tick-level aggregation
# ─────────────────────────────────────────────────────────────────────────── #

def _build_tick_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the multi-row-per-tick CSV into one row per (vehicleId, timestamp).

    Each tick row carries:
      - all master-side signals (constant across rows sharing the same tick)
      - the BEST candidate tower (highest candidateRSSI) seen that tick
      - the full candidate tower id list for that tick
      - selectedTower (constant across rows for the same tick)
    """
    agg_rows = []

    for (vid, ts), grp in df.groupby(["vehicleId", "timestamp"], sort=False):
        grp = grp.reset_index(drop=True)

        # Master-side values — take from first row (same for all rows of this tick)
        first = grp.iloc[0]

        # Full candidate set = all unique candidateMasterId values at this tick
        # plus the master (in case it doesn't appear as a candidate)
        master_id = int(safe_get(first, "masterId", 0))
        cand_ids  = grp["candidateMasterId"].dropna().astype(int).unique().tolist()
        if master_id not in cand_ids:
            cand_ids = [master_id] + cand_ids
        else:
            # Always put master first for consistent indexing
            cand_ids = [master_id] + [c for c in cand_ids if c != master_id]

        # Best candidate row = highest candidateRSSI (strongest signal seen this tick)
        best_idx  = grp["candidateRSSI"].idxmax()
        best_row  = grp.loc[best_idx]

        # selectedTower:  0 = stay on master = label index 0
        sel = int(safe_get(first, "selectedTower", 0))
        if sel == 0:
            sel = master_id   # normalise: 0 → masterId

        agg_rows.append({
            "vehicleId":         vid,
            "timestamp":         ts,
            "masterId":          master_id,
            "bestCandidateId":   int(safe_get(best_row, "candidateMasterId", master_id)),
            "candidateIds":      cand_ids,       # list stored as object
            "selectedTower":     sel,
            # Master signals
            "masterDistance":    safe_get(first, "masterDistance"),
            "masterRSSI":        safe_get(first, "masterRSSI"),
            "masterSINR":        safe_get(first, "masterSINR"),
            "masterRSRP":        safe_get(first, "masterRSRP"),
            "masterSpeed":       safe_get(first, "masterSpeed"),
            "towerload":         safe_get(first, "towerload"),
            # Best-candidate signals
            "candidateDistance": safe_get(best_row, "candidateDistance"),
            "candidateRSSI":     safe_get(best_row, "candidateRSSI"),
            "candidateSINR":     safe_get(best_row, "candidateSINR"),
            "candidateRSRP":     safe_get(best_row, "candidateRSRP"),
            "candidateSpeed":    safe_get(best_row, "candidateSpeed"),
            # Position / direction
            "vehicleDirection":  safe_get(first, "vehicleDirection"),
            "vehiclePosition-x": safe_get(first, "vehiclePosition-x"),
            "vehiclePosition-y": safe_get(first, "vehiclePosition-y"),
        })

    tick_df = pd.DataFrame(agg_rows)
    tick_df = tick_df.sort_values(["vehicleId", "timestamp"]).reset_index(drop=True)
    return tick_df


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset
# ─────────────────────────────────────────────────────────────────────────── #

class HandoverGraphSequenceDataset(Dataset):
    """
    Each sample:
      - seq_len consecutive tick-level graph snapshots for one vehicle
      - target = index of selectedTower in the candidate set at the label tick

    Label tick = the tick immediately after the sequence window.
    """

    def __init__(
        self,
        csv_path:     str,
        seq_len:      int = 10,
        pred_horizon: int = 1,
        split:        str = "train",
        train_ratio:  float = 0.70,
        val_ratio:    float = 0.15,
        standardizer: Optional[Standardizer] = None,
        random_seed:  int = 42,
    ):
        super().__init__()
        self.csv_path     = csv_path
        self.seq_len      = seq_len
        self.pred_horizon = pred_horizon
        self.split        = split
        self._fallback_count = 0
        self._total_count    = 0

        # ── Load and normalise column names ──────────────────────────────── #
        raw = pd.read_csv(csv_path)
        raw.columns = (
            raw.columns
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace("- ", "-", regex=False)
            .str.replace(" -", "-", regex=False)
        )
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        # ── Collapse multi-row-per-tick into one row per tick ─────────────── #
        logger.info("Building tick table from %d CSV rows...", len(raw))
        tick_df = _build_tick_table(raw)
        logger.info("Tick table: %d ticks, %d vehicles",
                    len(tick_df), tick_df["vehicleId"].nunique())

        self.feature_cols_to_scale = [
            "masterDistance", "candidateDistance",
            "masterRSSI", "candidateRSSI",
            "masterSINR", "candidateSINR",
            "masterRSRP", "candidateRSRP",
            "masterSpeed", "candidateSpeed",
            "vehiclePosition-x", "vehiclePosition-y",
            "towerload",
        ]

        # ── Vehicle-based train/val/test split ────────────────────────────── #
        all_vids = sorted(tick_df["vehicleId"].unique())
        rng      = np.random.default_rng(random_seed)
        rng.shuffle(all_vids)

        n         = len(all_vids)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        if split == "train":
            selected = set(all_vids[:train_end])
        elif split == "val":
            selected = set(all_vids[train_end:val_end])
        elif split == "test":
            selected = set(all_vids[val_end:])
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.split_df = tick_df[tick_df["vehicleId"].isin(selected)].copy()

        # Fit standardizer on training ticks only
        if standardizer is None:
            train_ticks       = tick_df[tick_df["vehicleId"].isin(set(all_vids[:train_end]))].copy()
            self.standardizer = fit_standardizer(train_ticks, self.feature_cols_to_scale)
        else:
            self.standardizer = standardizer

        # ── Build per-vehicle tick sequences ──────────────────────────────── #
        self.vehicle_groups = []
        for vid, grp in self.split_df.groupby("vehicleId"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            if len(grp) >= seq_len + pred_horizon:
                self.vehicle_groups.append((vid, grp))

        self.index_map = []
        for gi, (_, grp) in enumerate(self.vehicle_groups):
            max_start = len(grp) - seq_len - pred_horizon + 1
            for start in range(max_start):
                self.index_map.append((gi, start))

        logger.info("Split=%s  vehicles=%d  samples=%d",
                    split, len(self.vehicle_groups), len(self.index_map))

    # ── Snapshot builder ──────────────────────────────────────────────────── #

    def _build_snapshot(self, row: pd.Series) -> Data:
        sc = self.standardizer.transform   # shorthand

        ue_speed = sc("masterSpeed",       safe_get(row, "masterSpeed"))
        ue_x     = sc("vehiclePosition-x", safe_get(row, "vehiclePosition-x"))
        ue_y     = sc("vehiclePosition-y", safe_get(row, "vehiclePosition-y"))
        dir_sin, dir_cos = normalize_angle_deg(safe_get(row, "vehicleDirection"))

        m_dist = sc("masterDistance",   safe_get(row, "masterDistance"))
        m_rssi = sc("masterRSSI",       safe_get(row, "masterRSSI"))
        m_sinr = sc("masterSINR",       safe_get(row, "masterSINR"))
        m_rsrp = sc("masterRSRP",       safe_get(row, "masterRSRP"))
        m_load = sc("towerload",        safe_get(row, "towerload"))

        c_dist = sc("candidateDistance", safe_get(row, "candidateDistance"))
        c_rssi = sc("candidateRSSI",     safe_get(row, "candidateRSSI"))
        c_sinr = sc("candidateSINR",     safe_get(row, "candidateSINR"))
        c_rsrp = sc("candidateRSRP",     safe_get(row, "candidateRSRP"))

        master_id = int(safe_get(row, "masterId", 0))
        cand_id   = int(safe_get(row, "bestCandidateId", master_id))

        x = torch.tensor([
            # UE node
            [ue_speed, dir_sin, dir_cos, ue_x, ue_y, 0., 0., 0., 0., 0., 1., 0.],
            # Serving tower
            [0., 0., 0., 0., 0., m_dist, m_rssi, m_sinr, m_rsrp, m_load, 0., 1.],
            # Best candidate tower
            [0., 0., 0., 0., 0., c_dist, c_rssi, c_sinr, c_rsrp, 0., 0., 1.],
        ], dtype=torch.float)

        edge_index = torch.tensor(
            [[0,1, 0,2, 1,2, 2,1],
             [1,0, 2,0, 2,1, 1,2]], dtype=torch.long)

        edge_attr = torch.tensor([
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

    # ── Sample retrieval ──────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict:
        gi, start = self.index_map[idx]
        _, grp    = self.vehicle_groups[gi]

        seq_rows   = grp.iloc[start : start + self.seq_len]
        target_row = grp.iloc[start + self.seq_len + self.pred_horizon - 1]

        graph_sequence = [self._build_snapshot(r) for _, r in seq_rows.iterrows()]

        # ── Label: index of selectedTower in the label tick's candidate set ── #
        # The candidate set at the label tick = all towers seen for that tick.
        # selectedTower = masterId means "stay" = label index 0.
        sel          = int(safe_get(target_row, "selectedTower", 0))
        cand_ids_raw = target_row.get("candidateIds", None)

        # Reconstruct candidate id list for this tick
        if cand_ids_raw is not None and isinstance(cand_ids_raw, list):
            cand_ids = cand_ids_raw
        else:
            master_id = int(safe_get(target_row, "masterId", 0))
            best_cand = int(safe_get(target_row, "bestCandidateId", master_id))
            cand_ids  = [master_id, best_cand] if master_id != best_cand else [master_id]

        self._total_count += 1

        if sel in cand_ids:
            target_index = cand_ids.index(sel)
        else:
            # selectedTower is not in this tick's candidate set.
            # Happens when the CSV's selectedTower refers to a tower that wasn't
            # a candidate at the label tick (e.g. a handover to a third tower).
            # Fall back to 0 (stay on master) and log it.
            self._fallback_count += 1
            if self._fallback_count <= 20 or self._fallback_count % 500 == 0:
                logger.warning(
                    "selectedTower %d not in full candidate set %s at sample %d "
                    "(fallbacks: %d / %d)",
                    sel, cand_ids, idx,
                    self._fallback_count, self._total_count,
                )
            target_index = 0

        # Candidate tower ids for inference (master + best candidate from label tick)
        out_cand_ids = cand_ids[:2] if len(cand_ids) >= 2 else (cand_ids + cand_ids)[:2]

        return {
            "graph_sequence":      graph_sequence,
            "target_index":        torch.tensor(target_index, dtype=torch.long),
            "candidate_tower_ids": torch.tensor(out_cand_ids[:2], dtype=torch.long),
            "selected_tower":      torch.tensor(sel, dtype=torch.long),
        }

    def fallback_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._fallback_count / self._total_count
