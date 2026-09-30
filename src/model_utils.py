from __future__ import annotations

from pathlib import Path

import torch

from src.train_model import build_model


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet50") if isinstance(checkpoint, dict) else "resnet50"
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name