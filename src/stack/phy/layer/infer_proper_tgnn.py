import json
import logging
import sys
import torch

from graph_dataset import HandoverGraphSequenceDataset, Standardizer
from train_proper_tgnn import ProperTGNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_standardizer(path: str) -> Standardizer:
    with open(path, "r") as f:
        payload = json.load(f)
    return Standardizer(mean=payload["mean"], std=payload["std"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FIX: accept the runtime window CSV as a CLI argument so C++ can pass
    #      "runtime_tgnn_window.csv" and we infer on live data.
    #      Falls back to training CSV if no argument given (standalone runs).
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "simulator_data.csv"
    logger.info("Loading data from: %s", csv_path)

    standardizer = load_standardizer("proper_tgnn_ckpt/standardizer.json")

    ds = HandoverGraphSequenceDataset(
        csv_path=csv_path,
        seq_len=10,
        pred_horizon=1,
        split="test",
        standardizer=standardizer,
    )

    if len(ds) == 0:
        logger.error("Test dataset is empty — check csv_path and split sizes.")
        return

    model = ProperTGNN(node_dim=12, edge_dim=5, hidden_dim=64).to(device)
    model.load_state_dict(
        torch.load("proper_tgnn_ckpt/best_model.pt", map_location=device)
    )
    model.eval()

    sample          = ds[0]
    graph_sequence  = sample["graph_sequence"]
    candidate_ids   = sample["candidate_tower_ids"]

    for g in graph_sequence:
        g.x            = g.x.to(device)
        g.edge_index   = g.edge_index.to(device)
        g.edge_attr    = g.edge_attr.to(device)
        g.ue_index     = g.ue_index.to(device)
        g.tower_indices = g.tower_indices.to(device)

    with torch.no_grad():
        scores   = model(graph_sequence)
        pred_idx = int(torch.argmax(scores).item())

    logger.info("Candidate tower IDs : %s", candidate_ids.tolist())
    logger.info("Scores              : %s", [round(float(x), 4) for x in scores.cpu().tolist()])
    logger.info("Predicted tower     : %d", int(candidate_ids[pred_idx].item()))

    # Write C++-readable output: one line per tower → "towerId,score"
    output_path = "outputTGNN_proper.txt"
    with open(output_path, "w") as f:
        for tower_id, score in zip(candidate_ids.tolist(), scores.cpu().tolist()):
            f.write(f"{int(tower_id)},{float(score):.6f}\n")

    logger.info("Scores written to %s", output_path)


if __name__ == "__main__":
    main()
