import torch
import torch.nn as nn
import numpy as np

# =====================================================
# 1. Load training data (from OMNeT++)
# =====================================================
train_path = (
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
    "simu5G/src/stack/phy/layer/inputTGNN.txt"
)

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
X = X.unsqueeze(-1).unsqueeze(-1)   # (samples, time, nodes=1, features=1)
y = y.unsqueeze(-1)

# =====================================================
# 3. Graph definition (GCN)
# =====================================================
num_nodes = 1

# Identity adjacency (1-node graph for now)
A = torch.eye(num_nodes)

# Normalize adjacency
D = torch.diag(torch.sum(A, dim=1))
D_inv_sqrt = torch.inverse(torch.sqrt(D))
A_hat = D_inv_sqrt @ A @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, X, A_hat):
        # X: (batch, nodes, features)
        return torch.matmul(A_hat, self.W(X))


# =====================================================
# 4. GCN-LSTM Model
# =====================================================
class GCN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNLayer(1, 8)
        self.lstm = nn.LSTM(8, 20, batch_first=True)
        self.fc = nn.Linear(20, 1)

    def forward(self, X, A_hat):
        # X: (batch, time, nodes, features)
        batch, time, nodes, feats = X.shape

        gcn_out_seq = []

        for t in range(time):
            Xt = X[:, t, :, :]               # (batch, nodes, features)
            gcn_out = self.gcn(Xt, A_hat)    # (batch, nodes, 8)
            gcn_out_seq.append(gcn_out.squeeze(1))

        gcn_out_seq = torch.stack(gcn_out_seq, dim=1)
        # (batch, time, 8)

        _, (h, _) = self.lstm(gcn_out_seq)
        return self.fc(h[-1])


model = GCN_LSTM()

# =====================================================
# 5. Train (ONLINE, same philosophy as predLSTM.py)
# =====================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 25
for _ in range(epochs):
    optimizer.zero_grad()
    output = model(X, A_hat)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# =====================================================
# 6. Load test data
# =====================================================
test_path = (
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
    "simu5G/src/stack/phy/layer/inputTGNNTestData.txt"
)

test_data = np.loadtxt(test_path)
test_data = torch.tensor(test_data[:n_steps], dtype=torch.float32)

test_data = test_data.view(1, n_steps, 1, 1)

# =====================================================
# 7. Predict
# =====================================================
model.eval()
with torch.no_grad():
    prediction = model(test_data, A_hat).item()

# =====================================================
# 8. Write output for OMNeT++
# =====================================================
out_path = (
    "/home/ritika/Downloads/Ritika_Project/Project_GCN_LSTM_HO/"
    "simu5G/src/stack/phy/layer/outputTGNN.txt"
)

with open(out_path, "w") as f:
    f.write(str(prediction))
