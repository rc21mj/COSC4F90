"""
compare_models.py
=================
Performance comparison between TGNN and LSTM handover models.

Two complementary data sources are combined:

  1. performance_summary.csv  — written by LtePhyUe::performanceAnalysis()
     Contains per-model counters tracked live inside the OMNeT++ simulator
     (PDR, PLR, handovers, ping-pong, throughput) for BOTH models in a
     single run.

  2. simulator_data.csv  — the full per-row event log written by LtePhyUe
     Used to compute per-vehicle breakdowns and produce scatter plots.

Metrics reported
----------------
  1. Packet Delivery Ratio (PDR)   / Packet Loss Rate (PLR)
  2. Number of Handovers
  3. Number of Ping-Pong Handovers
  4. Average Throughput (Mbps)  — Shannon capacity proxy, 20 MHz, RSRP-based

Usage
-----
  python3 compare_models.py
  python3 compare_models.py --csv simulator_data.csv
  python3 compare_models.py --csv simulator_data.csv \\
      --summary performance_summary.csv --out results/

Outputs
-------
  comparison_summary.csv       aggregate metrics (per-vehicle mean +/- std)
  comparison_per_vehicle.csv   one row per vehicle x model
  comparison_plots.png         bar charts + scatter for all 5 metrics
  (stdout)                     formatted comparison table
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────── #
# Constants (must match LtePhyUe.cc)
# ─────────────────────────────────────────────────────────────────────────── #
RSRP_LOSS_THRESHOLD_DBM = -95.0
PING_PONG_WINDOW        = 5
BANDWIDTH_HZ            = 20e6

METRIC_LABELS = {
    "pdr":             "Packet Delivery Ratio",
    "plr":             "Packet Loss Rate",
    "handovers":       "# Handovers",
    "ping_pong":       "# Ping-Pong HOs",
    "throughput_mbps": "Avg Throughput (Mbps)",
}
MODEL_COLORS = {"tgnn": "#4C72B0", "lstm": "#DD8452", "ground": "#55A868"}
MODEL_LABELS = {"tgnn": "TGNN (Ours)", "lstm": "LSTM (Baseline)", "ground": "Ground Truth"}
LOWER_IS_BETTER = {"plr", "handovers", "ping_pong"}


# ─────────────────────────────────────────────────────────────────────────── #
# Throughput helper
# ─────────────────────────────────────────────────────────────────────────── #
def rsrp_to_throughput_mbps(rsrp_array: np.ndarray) -> np.ndarray:
    """
    Shannon capacity proxy: 20 MHz x log2(1 + SINR_linear)
    RSRP clamped to valid range [-120, -44] dBm.
    SINR clamped to realistic NR range [-10, 30] dB.
    Noise floor: -100 dBm for 20 MHz NR channel.
    """
    rsrp_clamped = np.clip(rsrp_array, -120.0, -44.0)
    sinr_db      = np.clip(rsrp_clamped + 100.0, -10.0, 30.0)
    sinr_lin     = np.power(10.0, sinr_db / 10.0).clip(min=0.001)
    return (BANDWIDTH_HZ * np.log2(1.0 + sinr_lin)) / 1e6


# ─────────────────────────────────────────────────────────────────────────── #
# Load performance_summary.csv (written by LtePhyUe::performanceAnalysis)
# ─────────────────────────────────────────────────────────────────────────── #
def load_simulator_summary(path: str) -> pd.DataFrame:
    """
    Load performance_summary.csv produced by the C++ simulator.
    Expected columns:
        model, pdr, plr, handovers, ping_pong, throughput_mbps,
        total_vehicles, intra_ho, inter_ho, failed_ho, ping_pong_global

    A row is flagged as suspect (no-inference run) only when ALL of:
      - handovers == 0
      - pdr == 1.0 exactly
      - throughput_mbps > 400
    That specific combination means TGNN never fired (no outputTGNN_proper.txt).
    A TGNN row with real handovers but high throughput is NOT suspect.
    """
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["model"] = df["model"].str.strip().str.lower()

        if "handovers" in df.columns and "throughput_mbps" in df.columns and "pdr" in df.columns:
            suspect = (
                (df["handovers"] == 0)
                & (df["pdr"].round(6) == 1.0)
                & (df["throughput_mbps"] > 400)
            )
            if suspect.any():
                print(f"[WARN] {suspect.sum()} suspect row(s) in {path} dropped "
                      f"(0 handovers + PDR=1.0 + high throughput = "
                      f"no-inference run).", file=sys.stderr)
                df = df[~suspect].copy()

        if df.empty:
            print("[WARN] All rows in summary were suspect. "
                  "Re-run with outputTGNN_proper.txt present.", file=sys.stderr)
            return pd.DataFrame()

        # Keep only the last row per model (most recent run wins)
        df = df.groupby("model", as_index=False).last()
        return df
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────── #
# Load and normalise simulator_data.csv
# ─────────────────────────────────────────────────────────────────────────── #
REQUIRED_COLS = [
    "timestamp", "vehicleId", "masterId", "candidateMasterId",
    "selectedTower", "predictedTGNN", "predictedLSTM",
    "masterRSSI", "candidateRSSI",
    "masterRSRP", "candidateRSRP",
    "masterSINR", "candidateSINR",
]

def load_event_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("- ", "-", regex=False)
        .str.replace(" -", "-", regex=False)
    )
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"simulator_data.csv missing columns: {missing}")
    df = df.sort_values(["vehicleId", "timestamp"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────── #
# Per-vehicle metric computation from event log
# ─────────────────────────────────────────────────────────────────────────── #
def _tower_rsrp(grp: pd.DataFrame, model: str) -> np.ndarray:
    """
    Return RSRP of the tower each model selected at each timestep.

    TGNN: uses selectedTower column — TGNN now drives the actual simulator
          decision. selectedTower=0 means no handover was made this tick,
          so the vehicle stays on masterId → use masterRSRP.

    LSTM: predictedLSTM is a dB-scale prediction, compare against candidateRSSI.

    Ground truth: same as TGNN (selectedTower column).
    """
    if model in ("tgnn", "ground"):
        sel    = grp["selectedTower"].values
        master = grp["masterId"].values
        # selectedTower==0 means no HO decision → vehicle stays on current master
        on_master = (sel == 0) | (sel == master)
        return np.where(on_master, grp["masterRSRP"].values, grp["candidateRSRP"].values)
    else:  # lstm
        pick_master = grp["predictedLSTM"].values > grp["candidateRSSI"].values
        return np.where(pick_master, grp["masterRSRP"].values, grp["candidateRSRP"].values)


def _tower_ids(grp: pd.DataFrame, model: str) -> np.ndarray:
    """
    Return tower ID selected by each model at each timestep.
    selectedTower=0 → vehicle stays on masterId (no handover this tick).
    """
    if model in ("tgnn", "ground"):
        sel    = grp["selectedTower"].values.astype(int)
        master = grp["masterId"].values.astype(int)
        # Replace 0 (no-decision marker) with current masterId
        return np.where(sel == 0, master, sel)
    else:  # lstm
        pick_master = grp["predictedLSTM"].values > grp["candidateRSSI"].values
        return np.where(
            pick_master,
            grp["masterId"].values.astype(int),
            grp["candidateMasterId"].values.astype(int),
        )


def _pdr(rsrp: np.ndarray) -> float:
    return float(np.mean(rsrp > RSRP_LOSS_THRESHOLD_DBM))


def _handovers(tids: np.ndarray) -> int:
    if len(tids) < 2:
        return 0
    return int(np.sum(tids[1:] != tids[:-1]))


def _ping_pong(tids: np.ndarray, window: int = PING_PONG_WINDOW) -> int:
    if len(tids) < 2:
        return 0
    ho_idx = np.where(tids[1:] != tids[:-1])[0] + 1
    count  = 0
    for i in range(len(ho_idx) - 1):
        t1, t2 = ho_idx[i], ho_idx[i + 1]
        if (t2 - t1) <= window and tids[t2] == tids[t1 - 1]:
            count += 1
    return count


def _throughput(rsrp: np.ndarray) -> float:
    return float(np.mean(rsrp_to_throughput_mbps(rsrp)))


def compute_per_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vid, grp in df.groupby("vehicleId"):
        grp = grp.reset_index(drop=True)
        row = {"vehicleId": vid, "n_steps": len(grp)}
        for model in ["tgnn", "lstm", "ground"]:
            rsrp = _tower_rsrp(grp, model)
            tids = _tower_ids(grp, model)
            row[f"{model}_pdr"]             = _pdr(rsrp)
            row[f"{model}_plr"]             = 1.0 - _pdr(rsrp)
            row[f"{model}_handovers"]       = _handovers(tids)
            row[f"{model}_ping_pong"]       = _ping_pong(tids)
            row[f"{model}_throughput_mbps"] = _throughput(rsrp)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(pv: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pdr", "plr", "handovers", "ping_pong", "throughput_mbps"]
    rows = []
    for m in metrics:
        row = {"metric": m}
        for model in ["tgnn", "lstm", "ground"]:
            col = f"{model}_{m}"
            row[f"{model}_mean"] = pv[col].mean()
            row[f"{model}_std"]  = pv[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────── #
# Merge simulator summary into aggregate
# ─────────────────────────────────────────────────────────────────────────── #
def merge_with_simulator(agg: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """
    Override CSV-derived means with direct C++ counters where available.
    Uses the first (and after deduplication, only) row per model.
    """
    if sim.empty:
        return agg

    agg = agg.copy()
    metric_map = {
        "pdr":             "pdr",
        "plr":             "plr",
        "handovers":       "handovers",
        "ping_pong":       "ping_pong",
        "throughput_mbps": "throughput_mbps",
    }

    # After load_simulator_summary deduplication there should be exactly
    # one row per model — iterate and use each one.
    for model in ("tgnn", "lstm"):
        model_rows = sim[sim["model"] == model]
        if model_rows.empty:
            continue
        sim_row = model_rows.iloc[0]   # one row after dedup
        for metric_key, sim_col in metric_map.items():
            if sim_col in sim_row and not pd.isna(sim_row[sim_col]):
                mask = agg["metric"] == metric_key
                agg.loc[mask, f"{model}_mean"] = float(sim_row[sim_col])
                agg.loc[mask, f"{model}_std"]  = 0.0
    return agg


# ─────────────────────────────────────────────────────────────────────────── #
# Plotting
# ─────────────────────────────────────────────────────────────────────────── #
def plot_all(agg: pd.DataFrame, pv: pd.DataFrame, out_path: str,
             has_simulator_data: bool = False):
    metrics = ["pdr", "plr", "handovers", "ping_pong", "throughput_mbps"]
    models  = ["tgnn", "lstm", "ground"]

    fig = plt.figure(figsize=(20, 12))
    suffix = " — C++ counters + CSV" if has_simulator_data else " — CSV-derived"
    fig.suptitle(
        "TGNN vs LSTM  |  Handover Performance Comparison" + suffix,
        fontsize=14, fontweight="bold", y=0.99,
    )

    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]

    x     = np.arange(len(models))
    width = 0.50

    for ax, metric in zip(axes, metrics):
        row    = agg[agg["metric"] == metric].iloc[0]
        means  = [float(row[f"{m}_mean"]) for m in models]
        stds   = [float(row[f"{m}_std"])  for m in models]
        colors = [MODEL_COLORS[m] for m in models]

        bars = ax.bar(
            x, means, width, yerr=stds, capsize=5,
            color=colors, edgecolor="white", linewidth=0.8, alpha=0.88,
            error_kw={"ecolor": "#888", "elinewidth": 1.2},
        )

        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8.5, rotation=10)
        ax.set_ylabel("Mean ± Std" if max(stds) > 0 else "Value", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        max_std = max(stds) if max(stds) > 0 else max(means) * 0.02
        for bar, mean in zip(bars, means):
            fmt = f"{mean:.3f}" if abs(mean) < 100 else f"{mean:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_std * 0.08 + 1e-6,
                fmt, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

        lower = metric in LOWER_IS_BETTER
        best  = int(np.argmin(means) if lower else np.argmax(means))
        bars[best].set_edgecolor("#2ca02c")
        bars[best].set_linewidth(2.8)

    # Scatter: handovers vs ping-pong per vehicle
    ax_sc = fig.add_subplot(gs[1, 2])
    for model in ["tgnn", "lstm"]:
        ax_sc.scatter(
            pv[f"{model}_handovers"], pv[f"{model}_ping_pong"],
            alpha=0.55, s=22, color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )
    ax_sc.set_xlabel("Handovers per vehicle", fontsize=9)
    ax_sc.set_ylabel("Ping-Pong HOs per vehicle", fontsize=9)
    ax_sc.set_title("HO vs Ping-Pong (per vehicle)", fontsize=11, fontweight="bold", pad=8)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    ax_sc.legend(fontsize=8.5, framealpha=0.7)
    ax_sc.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_sc.set_axisbelow(True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────── #
# Console tables
# ─────────────────────────────────────────────────────────────────────────── #
def print_simulator_counters(sim: pd.DataFrame):
    if sim.empty:
        return
    print("\n" + "=" * 76)
    print("  SIMULATOR HANDOVER COUNTERS  (LtePhyUe::performanceAnalysis)")
    print("=" * 76)
    show = [c for c in [
        "model", "total_vehicles",
        "intra_ho", "inter_ho", "failed_ho", "ping_pong_global",
        "pdr", "plr", "handovers", "ping_pong", "throughput_mbps",
    ] if c in sim.columns]
    print(sim[show].to_string(index=False))
    print("=" * 76 + "\n")


def print_table(agg: pd.DataFrame, has_sim: bool = False):
    source = " [source: C++ simulator]" if has_sim else " [source: CSV-derived]"
    print("\n" + "=" * 76)
    print(f"  TGNN vs LSTM — Performance Comparison{source}")
    print("=" * 76)
    print(f"{'Metric':<28}{'TGNN':>12}{'LSTM':>12}{'Delta (T-L)':>14}{'Winner':>10}")
    print("-" * 76)

    for _, row in agg.iterrows():
        metric = row["metric"]
        t_mean = float(row["tgnn_mean"])
        l_mean = float(row["lstm_mean"])
        delta  = t_mean - l_mean
        lower  = metric in LOWER_IS_BETTER

        if abs(delta) < 1e-9:
            winner = "Tie"
        elif lower:
            winner = "TGNN ✓" if delta < 0 else "LSTM ✓"
        else:
            winner = "TGNN ✓" if delta > 0 else "LSTM ✓"

        fmt = ".4f" if abs(t_mean) < 100 else ".2f"
        print(
            f"{METRIC_LABELS[metric]:<28}"
            f"{t_mean:>{12}{fmt}}"
            f"{l_mean:>{12}{fmt}}"
            f"{delta:>{14}{fmt}}"
            f"{winner:>10}"
        )

    print("=" * 76)
    print("  Green border in plots = best performer per metric")
    print("  Throughput: Shannon capacity, 20 MHz, RSRP noise floor -97 dBm")
    print("  PDR/PLR threshold: RSRP > -95 dBm = packet delivered\n")


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point
# ─────────────────────────────────────────────────────────────────────────── #
def main():
    parser = argparse.ArgumentParser(
        description="TGNN vs LSTM handover performance comparison"
    )
    parser.add_argument("--csv",     default="simulator_data.csv",
                        help="Path to simulator_data.csv (event log)")
    parser.add_argument("--summary", default="",
                        help="Path to performance_summary.csv from LtePhyUe (optional)")
    parser.add_argument("--out",     default=".",
                        help="Output directory for CSVs and PNG plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. C++ simulator counters
    sim     = load_simulator_summary(args.summary)
    has_sim = not sim.empty

    if has_sim:
        print_simulator_counters(sim)
        sim.to_csv(os.path.join(args.out, "simulator_counters.csv"), index=False)
        print(f"[INFO] Simulator counters -> {os.path.join(args.out, 'simulator_counters.csv')}")
    else:
        if args.summary:
            print(f"[WARN] --summary not found: {args.summary}")
        print("[INFO] No simulator summary — computing all metrics from CSV")

    # 2. Per-vehicle metrics from event log
    print(f"[INFO] Loading event log: {args.csv}")
    df  = load_event_log(args.csv)
    print(f"[INFO] {len(df):,} rows, {df['vehicleId'].nunique()} vehicles")

    pv  = compute_per_vehicle(df)
    agg = aggregate(pv)

    # 3. Override means with C++ counters where available
    if has_sim:
        agg = merge_with_simulator(agg, sim)
        print("[INFO] C++ counters merged into aggregate (override CSV means for tgnn/lstm)")

    # 4. Save
    pv_path  = os.path.join(args.out, "comparison_per_vehicle.csv")
    agg_path = os.path.join(args.out, "comparison_summary.csv")
    pv.to_csv(pv_path,  index=False)
    agg.to_csv(agg_path, index=False)
    print(f"[INFO] Per-vehicle -> {pv_path}")
    print(f"[INFO] Summary     -> {agg_path}")

    # 5. Plot
    plot_all(agg, pv, os.path.join(args.out, "comparison_plots.png"), has_sim)

    # 6. Print table
    print_table(agg, has_sim)


if __name__ == "__main__":
    main()
