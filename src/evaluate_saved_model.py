from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.dataset import create_dataloaders
from src.train_model import build_model, compute_metrics

ROOT_DIR = Path(__file__).resolve().parent.parent


def evaluate_checkpoint(model_path: str | Path, metadata_path: str | Path, batch_size: int = 32, num_workers: int = 4, device_name: str | None = None) -> dict:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = Path(model_path)
    payload = torch.load(model_path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        model_name = payload.get("model_name", "convnext_tiny")
        state_dict = payload["state_dict"]
    else:
        model_name = "convnext_tiny"
        state_dict = payload

    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _, _, test_loader = create_dataloaders(metadata_path=Path(metadata_path), batch_size=batch_size, num_workers=num_workers)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    y_true, y_prob = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(test_loader.dataset)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation d'un modele de deepfake detection sauvegarde.")
    parser.add_argument("--model-path", type=str, default=str(ROOT_DIR / "exp_convnext" / "best_model.pth"))
    parser.add_argument("--metadata-path", type=str, default=str(ROOT_DIR / "data" / "processed" / "metadata.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate_checkpoint(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))
