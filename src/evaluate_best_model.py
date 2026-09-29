from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset import create_dataloaders
from src.train_model import DEFAULT_METADATA_PATH, build_model, compute_metrics

ROOT_DIR = Path(__file__).resolve().parent.parent


def save_confusion_matrix(y_true, y_pred, save_path: Path) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps({"labels": ["Real", "Fake"], "confusion_matrix": cm.tolist()}, indent=2),
        encoding="utf-8",
    )
    save_path.with_suffix(".csv").write_text(np.array2string(cm, separator=","), encoding="utf-8")


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    model_name: str,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    output_dir: str | Path = ROOT_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    device_name: str | None = None,
):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _, _, test_loader = create_dataloaders(
        metadata_path=Path(metadata_path),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    y_true, y_prob = [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)
            logits = model(images).squeeze(1)
            total_loss += criterion(logits, labels).item() * images.size(0)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(test_loader.dataset)

    confusion_path = output_dir / f"{model_name}_confusion.json"
    save_confusion_matrix(y_true, (y_prob >= 0.5).astype(int), confusion_path)

    summary = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }
    summary_path = output_dir / f"{model_name}_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Evaluation du meilleur modele ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
    print(f"Matrice de confusion: {metrics['confusion_matrix']}")
    print(f"Summary: {summary_path}")
    print(f"Confusion JSON: {confusion_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluer un checkpoint sauvegarde sur le jeu de test.")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "best_model.pth"))
    parser.add_argument("--model", choices=["resnet50", "convnext_tiny"], default="resnet50")
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
