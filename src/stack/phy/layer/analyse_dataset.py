"""
analyse_dataset.py
==================
Understand why the HO rate can't be pushed below ~35% and what the
real handover frequency looks like in simulator_data.csv.

Run: python3 analyse_dataset.py
"""

import pandas as pd
import numpy as np

df = pd.read_csv("simulator_data.csv")
df.columns = (df.columns.str.strip()
                        .str.replace(r"\s+", " ", regex=True)
                        .str.replace("- ", "-", regex=False))

print(f"Total rows         : {len(df)}")
print(f"Unique vehicles    : {df['vehicleId'].nunique()}")
print(f"Unique timestamps  : {df['timestamp'].nunique()}")

# ── Filter out same-tower rows (same as graph_dataset.py)
df_real = df[df["masterId"] != df["candidateMasterId"]].copy()
print(f"\nAfter dropping masterId==candidateMasterId: {len(df_real)} rows")

# ── selectedTower distribution
stay   = (df_real["selectedTower"] == df_real["masterId"]).sum()
switch = (df_real["selectedTower"] != df_real["masterId"]).sum()
print(f"\nStay   (selectedTower == masterId) : {stay}  ({stay/len(df_real)*100:.1f}%)")
print(f"Switch (selectedTower != masterId) : {switch}  ({switch/len(df_real)*100:.1f}%)")

# ── Per-vehicle HO rate
print("\nPer-vehicle HO rate:")
for vid, grp in df_real.groupby("vehicleId"):
    n_switch = (grp["selectedTower"] != grp["masterId"]).sum()
    print(f"  Vehicle {vid}: {n_switch}/{len(grp)} = {n_switch/len(grp)*100:.1f}% HO")

# ── candidateRSSI vs masterRSSI delta distribution
df_real = df_real.copy()
df_real["rssi_delta"] = df_real["candidateRSSI"] - df_real["masterRSSI"]
df_real["sinr_delta"] = df_real["candidateSINR"] - df_real["masterSINR"]

print(f"\nRSSI delta (candidate - master):")
print(f"  mean={df_real['rssi_delta'].mean():.2f}  std={df_real['rssi_delta'].std():.2f}")
print(f"  min={df_real['rssi_delta'].min():.2f}  max={df_real['rssi_delta'].max():.2f}")

# Among switch decisions, how much better is candidate?
switches = df_real[df_real["selectedTower"] != df_real["masterId"]]
stays    = df_real[df_real["selectedTower"] == df_real["masterId"]]
print(f"\nAmong SWITCH rows - RSSI delta: mean={switches['rssi_delta'].mean():.2f}  std={switches['rssi_delta'].std():.2f}")
print(f"Among STAY rows   - RSSI delta: mean={stays['rssi_delta'].mean():.2f}  std={stays['rssi_delta'].std():.2f}")

# ── Key insight: how many switch rows have candidateRSSI only marginally better?
thresholds = [0, 1, 2, 3, 5, 8, 10]
print(f"\nSwitch rows where candidateRSSI - masterRSSI < threshold (these are questionable HOs):")
for t in thresholds:
    marginal = (switches["rssi_delta"] < t).sum()
    print(f"  < {t:2d} dB : {marginal}/{len(switches)} = {marginal/len(switches)*100:.1f}%")

print(f"\nIf we only switch when RSSI delta > 3 dB, remaining HO rate would be:")
would_switch = (df_real["rssi_delta"] > 3).sum()
print(f"  {would_switch}/{len(df_real)} = {would_switch/len(df_real)*100:.1f}%")

print(f"\nIf we only switch when RSSI delta > 5 dB, remaining HO rate would be:")
would_switch = (df_real["rssi_delta"] > 5).sum()
print(f"  {would_switch}/{len(df_real)} = {would_switch/len(df_real)*100:.1f}%")
