from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from src.dataset import create_dataloaders

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_PATH = ROOT_DIR / "data" / "processed" / "metadata.csv"
DEFAULT_BEST_MODEL_PATH = ROOT_DIR / "best_model.pth"


def build_model(model_name: str, pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """Construit l'architecture demandee et remplace la couche finale.

    Binary Classification: Real vs Fake -> sortie scalaire (logits) et BCEWithLogitsLoss.
    """
    if model_name.lower() == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name.lower() == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Architecture inconnue: {model_name}. Choisir 'resnet50' ou 'convnext_tiny'.")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcule les metricas d'evaluation pour la classification binaire."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics


def evaluate_model(model: nn.Module, dataloader, device: torch.device, criterion: nn.Module):
    """Evaluation sur le dataset validation/test, retourne loss et metrics."""
    model.eval()
    total_loss = 0.0
    outputs, targets = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probas = torch.sigmoid(logits).cpu().numpy()
            outputs.extend(probas.tolist())
            targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.asarray(targets), np.asarray(outputs))
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion, scheduler, device: torch.device, scaler):
    """Train sur une epoch complet avec Mixed Precision sur CUDA."""
    model.train()
    total_loss = 0.0
    outputs, targets = [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

        probas = torch.sigmoid(logits).detach().cpu().numpy()
        outputs.extend(probas.tolist())
        targets.extend(labels.detach().cpu().numpy().tolist())

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.asarray(targets), np.asarray(outputs))
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def save_best_state(model: nn.Module, path: Path, metrics: dict, model_name: str) -> None:
    payload = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)


def save_confusion_matrix(y_true, y_pred, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "labels": ["Real", "Fake"],
        "confusion_matrix": cm.tolist(),
    }
    save_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    csv_path = save_path.with_suffix(".csv")
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")


def train_and_evaluate(
    model_name: str = "resnet50",
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    output_dir: str | Path = ROOT_DIR,
    pretrained: bool = True,
    device_name: str | None = None,
):
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=Path(metadata_path),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(model_name=model_name, pretrained=pretrained)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(log_dir=str(output_dir / "logs" / model_name))

    best_auc = -1.0
    best_state_path = output_dir / "best_model.pth"
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
        )

        val_loss, val_metrics = evaluate_model(model=model, dataloader=val_loader, device=device, criterion=criterion)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/f1", val_metrics["f1"], epoch)
        writer.add_scalar("val/roc_auc", val_metrics["roc_auc"], epoch)

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_metrics = {**val_metrics, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            save_best_state(model, best_state_path, best_metrics, model_name)
            print(f"Nouveau meilleur modele sauvegarde: {best_state_path}")

    # Evaluation finale sur le test set avec les meilleurs poids
    final_test_loss, final_test_metrics = evaluate_model(model, test_loader, device, criterion)

    # Recalcul des predictions complètes sur le test set
    model.eval()
    y_true_test, y_prob_test = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            y_prob_test.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            y_true_test.extend(labels.numpy().tolist())

    test_metrics = compute_metrics(np.asarray(y_true_test), np.asarray(y_prob_test))
    test_metrics["loss"] = final_test_loss

    confusion_path = output_dir / "confusion_matrix_test.json"
    save_confusion_matrix(np.asarray(y_true_test), (np.asarray(y_prob_test) >= 0.5).astype(int), confusion_path)

    summary = {
        "model": model_name,
        "best_auc": best_auc,
        "best_model_path": str(best_state_path),
        "test_metrics": test_metrics,
        "best_validation_metrics": best_metrics,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    writer.add_scalar("test/loss", test_metrics["loss"], 1)
    writer.add_scalar("test/accuracy", test_metrics["accuracy"], 1)
    writer.add_scalar("test/f1", test_metrics["f1"], 1)
    writer.add_scalar("test/roc_auc", test_metrics["roc_auc"], 1)
    writer.flush()
    writer.close()

    print("\n=== Resultats finaux sur le test ===")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"AUC-ROC: {test_metrics['roc_auc']:.4f}")
    print(f"Matrice confusion: {test_metrics['confusion_matrix']}")
    print(f"Fichier summary: {summary_path}")
    print(f"Meilleur modele: {best_state_path}")

    return model, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement CNN pour la detection de deepfakes.")
    parser.add_argument("--model", choices=["resnet50", "convnext_tiny"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        pretrained=args.pretrained,
        device_name=args.device,
    )
