import json
import torch

from graph_dataset import HandoverGraphSequenceDataset, Standardizer
from train_proper_tgnn import ProperTGNN


def load_standardizer(path: str) -> Standardizer:
    with open(path, "r") as f:
        payload = json.load(f)
    return Standardizer(mean=payload["mean"], std=payload["std"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    standardizer = load_standardizer("proper_tgnn_ckpt/standardizer.json")

    ds = HandoverGraphSequenceDataset(
        csv_path="simulator_data.csv",
        seq_len=10,
        pred_horizon=1,
        split="test",
        standardizer=standardizer,
    )

    model = ProperTGNN(node_dim=12, edge_dim=5, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("proper_tgnn_ckpt/best_model.pt", map_location=device))
    model.eval()

    sample = ds[0]
    graph_sequence = sample["graph_sequence"]
    candidate_ids = sample["candidate_tower_ids"]

    for g in graph_sequence:
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        g.edge_attr = g.edge_attr.to(device)
        g.ue_index = g.ue_index.to(device)
        g.tower_indices = g.tower_indices.to(device)

    with torch.no_grad():
        scores = model(graph_sequence)
        pred_idx = int(torch.argmax(scores).item())

    print("Candidate tower IDs:", candidate_ids.tolist())
    print("Scores:", [float(x) for x in scores.cpu().tolist()])
    print("Predicted tower:", int(candidate_ids[pred_idx].item()))

    # Optional: write in a C++-friendly format
    with open("outputTGNN_proper.txt", "w") as f:
        for tower_id, score in zip(candidate_ids.tolist(), scores.cpu().tolist()):
            f.write(f"{int(tower_id)},{float(score):.6f}\n")


if __name__ == "__main__":
    main()