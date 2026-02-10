import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time

# =====================================================
# CONFIG (defaults; can be overridden via CLI args/env)
# =====================================================
N_STEPS = int(os.environ.get("TGNN_STEPS", "5"))
NUM_FEATURES = 4   # RSRP, SINR, SPEED, DISTANCE
HIDDEN_DIM = 32
EPOCHS = int(os.environ.get("TGNN_EPOCHS", "25"))
LR = float(os.environ.get("TGNN_LR", "0.001"))

# Default file paths (kept compatible with your current project layout)
DEFAULT_TRAIN_PATH = (
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
    "simu5G/src/stack/phy/layer/inputTGNNdiff.txt"
)
DEFAULT_TEST_PATH = (
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
    "simu5G/src/stack/phy/layer/inputTGNNdiffTestData.txt"
)
DEFAULT_OUT_PATH = (
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
    "simu5G/src/stack/phy/layer/outputTGNNdiff.txt"
)

# =====================================================
# LOAD DATA FROM OMNeT++
# Format per line:
# time node_id rsrp sinr speed distance
# =====================================================
def load_data(path: str) -> torch.Tensor:
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.shape[1] < 6:
        raise ValueError(f"Expected at least 6 columns, got {raw.shape[1]} in {path}")

    times = raw[:, 0]
    nodes = raw[:, 1].astype(int)

    unique_nodes = np.unique(nodes)
    unique_times = np.unique(times)

    # Map actual time values (can be non-integer) to 0..T-1
    time_map = {t: i for i, t in enumerate(unique_times)}
    node_map = {n: i for i, n in enumerate(unique_nodes)}

    T = len(unique_times)
    N = len(unique_nodes)
    F = NUM_FEATURES

    X = np.zeros((T, N, F), dtype=np.float32)

    for row in raw:
        t_val = row[0]
        n_val = int(row[1])
        ti = time_map.get(t_val, None)
        ni = node_map.get(n_val, None)
        if ti is None or ni is None:
            continue
        X[ti, ni] = row[2:2 + NUM_FEATURES]

    return torch.tensor(X, dtype=torch.float32)


# =====================================================
# BUILD DYNAMIC ADJACENCY (distance-based)
# =====================================================
def build_adjacency(X_nodes: torch.Tensor, threshold: float = 150.0) -> torch.Tensor:
    dist = X_nodes[:, -1]
    N = dist.shape[0]

    A = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            if abs(dist[i] - dist[j]) < threshold:
                A[i, j] = 1.0

    # self-loops
    A.fill_diagonal_(1.0)

    # Row-normalize (D^-1 A)
    D = torch.sum(A, dim=1)  # (N,)
    D_inv = torch.where(D > 0, 1.0 / D, torch.zeros_like(D))
    return D_inv.unsqueeze(1) * A


# =====================================================
# GRAPH ATTENTION LAYER (masked)
# =====================================================
class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = self.fc(X)
        N = H.size(0)

        scores = torch.full((N, N), -1e9, dtype=H.dtype, device=H.device)
        for i in range(N):
            for j in range(N):
                if A[i, j] > 0:
                    scores[i, j] = self.attn(torch.cat([H[i], H[j]])).squeeze(-1)

        alpha = torch.softmax(scores, dim=1)
        return alpha @ H


# =====================================================
# TGNN MODEL (supports batch)
# =====================================================
class TGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATLayer(NUM_FEATURES, HIDDEN_DIM)
        self.gru = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (T,N,F) or (B,T,N,F)
        if X.dim() == 3:
            X = X.unsqueeze(0)
        B, T, N, F = X.shape

        seqs = []
        for b in range(B):
            pooled = []
            for t in range(T):
                A = build_adjacency(X[b, t])
                h_nodes = self.gat(X[b, t], A)     # (N, H)
                pooled.append(h_nodes.mean(dim=0)) # (H,)
            seqs.append(torch.stack(pooled, dim=0)) # (T, H)

        seqs = torch.stack(seqs, dim=0) # (B, T, H)
        _, h = self.gru(seqs)          # (1, B, H)
        return self.fc(h[-1])          # (B, 1)


def make_sequences(X_all: torch.Tensor, n_steps: int):
    T = X_all.shape[0]
    if T <= n_steps:
        return None, None
    X_seq, y_seq = [], []
    for i in range(T - n_steps):
        X_seq.append(X_all[i:i + n_steps])
        y_seq.append(X_all[i + n_steps, :, 0].mean())  # mean RSRP at next step
    return torch.stack(X_seq, dim=0), torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)


def train_model(model: nn.Module, X_seq: torch.Tensor, y_seq: torch.Tensor):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n = X_seq.shape[0]
    batch_size = min(16, n)

    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_seq[idx]
            yb = y_seq[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()


def main():
    # Usage:
    #   python predTGNNdiff_fixed.py [train_path] [test_path] [out_path]
    train_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRAIN_PATH
    test_path  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEST_PATH
    out_path   = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT_PATH

    X_all = load_data(train_path)
    X_seq, y_seq = make_sequences(X_all, N_STEPS)

    model = TGNN()
    if X_seq is not None:
        train_model(model, X_seq, y_seq)
# you are a bully ritika :(
# YOUR BAD CODE HURTS ME
# This IDE IS ass too


    if X_test.shape[0] >= N_STEPS:
        X_test = X_test[-N_STEPS:]
    else:
        pad = X_test[0:1].repeat(N_STEPS - X_test.shape[0], 1, 1)
        X_test = torch.cat([pad, X_test], dim=0)

    model.eval()
    with torch.no_grad():
        pred = float(model(X_test).item())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(str(pred))
    
    with open("/tmp/tgnndiff_out.txt", "a") as f:
        f.write(f"{time.time()} {pred}\n")
        
    with open("/tmp/predTGNNdiff_ran.txt", "a") as f:
        f.write(f"predTGNNdiff STARTED at {time.ctime()}\n")


if __name__ == "__main__":
    main()