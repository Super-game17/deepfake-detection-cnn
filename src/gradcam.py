from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models

from src.augmentations import get_eval_transform
from src.model_utils import load_checkpoint


ROOT_DIR = Path(__file__).resolve().parent.parent


class RealClassTarget:
    def __call__(self, model_output):
        return -model_output[:, 0]


def generate_gradcam(
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    device_name: str | None = None,
    target_label: str = "auto",
):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model, _ = load_checkpoint(checkpoint_path, device)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image impossible a lire: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = get_eval_transform()(image=image_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()
    predicted_label = "fake" if probability >= 0.5 else "real"
    selected_label = predicted_label if target_label == "auto" else target_label
    targets = [
        ClassifierOutputTarget(0)
        if selected_label == "fake"
        else RealClassTarget()
    ]

    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    base = cv2.resize(image_bgr, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, heatmap, 0.45, 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return {
        "image": str(image_path),
        "output": str(output_path),
        "fake_probability": probability,
        "real_probability": 1.0 - probability,
        "prediction": predicted_label,
        "target": selected_label,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générer une heatmap Grad-CAM")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT_DIR / "best_model.pth"))
    parser.add_argument("--output", default=str(ROOT_DIR / "outputs" / "gradcam_result.jpg"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--target", choices=["auto", "real", "fake"], default="auto")
    args = parser.parse_args()
    print(generate_gradcam(args.image, args.checkpoint, args.output, args.device, args.target))