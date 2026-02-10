import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================
# 1. Load training data (from OMNeT++)
# =====================================================
train_path = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/inputTGNN.txt"
train_data = np.loadtxt(train_path)
train_data = torch.tensor(train_data, dtype=torch.float32)

# =====================================================
# 2. Temporal windowing
# =====================================================
def split_sequence(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return torch.stack(X), torch.stack(y)

n_steps = 5
X, y = split_sequence(train_data, n_steps)

# Shapes:
# X → (samples, time)
# y → (samples)

# Add node + feature dimensions
num_nodes = 1
node_features = 1
X = X.unsqueeze(-1).unsqueeze(-1)   # (samples, time, nodes, features)
y = y.unsqueeze(-1)

# =====================================================
# 3. TemporalGNN Model
# =====================================================
# Note: requires torch_geometric for GConvGRU
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.data import Data

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, out_channels, K):
        super(TemporalGNN, self).__init__()
        self.gconvgru = GConvGRU(in_channels=node_features, out_channels=hidden_channels, K=K)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: (batch, time, nodes, features)
        batch, time, nodes, feats = x.shape
        h_seq = []
        for t in range(time):
            xt = x[:, t, :, :]  # (batch, nodes, features)
            xt = xt.squeeze(-1) # (batch, nodes)
            # GConvGRU expects (nodes, features) per graph
            h = []
            for b in range(batch):
                h_b = self.gconvgru(xt[b], edge_index)
                h.append(h_b)
            h = torch.stack(h)  # (batch, nodes, hidden)
            h_seq.append(h.mean(dim=1))  # Aggregate nodes
        h_seq = torch.stack(h_seq, dim=1)  # (batch, time, hidden)
        out = self.linear(F.relu(h_seq[:, -1, :]))
        return out

hidden_channels = 8
out_channels = 1
K = 2  # Chebyshev polynomial order (or hops)
model = TemporalGNN(node_features, hidden_channels, out_channels, K)

# =====================================================
# 4. Edge index (fully connected 1-node graph for now)
# =====================================================
# For multiple nodes, define proper edge_index
edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # (2, num_edges)

# =====================================================
# 5. Train
# =====================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 25
for _ in range(epochs):
    optimizer.zero_grad()
    output = model(X, edge_index)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# =====================================================
# 6. Load test data
# =====================================================
test_path = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/inputTGNNTestData.txt"
test_data = np.loadtxt(test_path)
test_data = torch.tensor(test_data[:n_steps], dtype=torch.float32)
test_data = test_data.view(1, n_steps, num_nodes, node_features)

# =====================================================
# 7. Predict
# =====================================================
model.eval()
with torch.no_grad():
    prediction = model(test_data, edge_index).item()

# =====================================================
# 8. Write output for OMNeT++
# =====================================================
out_path = "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/simu5G/src/stack/phy/layer/outputTGNN.txt"
with open(out_path, "w") as f:
    f.write(str(prediction))
