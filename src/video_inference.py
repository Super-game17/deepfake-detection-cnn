from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

from src.augmentations import get_eval_transform
from src.model_utils import load_checkpoint


ROOT_DIR = Path(__file__).resolve().parent.parent
FRAME_INTERVAL = 15
MIN_CONFIDENCE = 0.50
MIN_FACE_SIZE = 60


def create_scrfd_detector():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA est obligatoire pour l'inférence vidéo SCRFD.")
    detector = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    return detector


def crop_faces(frame, detector):
    crops = []
    for face in detector.get(frame):
        x1, y1, x2, y2 = face.bbox.astype(int)
        width, height = x2 - x1, y2 - y1
        if face.det_score < MIN_CONFIDENCE or min(width, height) < MIN_FACE_SIZE:
            continue
        margin_x, margin_y = int(width * 0.1), int(height * 0.1)
        crop = frame[
            max(0, y1 - margin_y):min(frame.shape[0], y2 + margin_y),
            max(0, x1 - margin_x):min(frame.shape[1], x2 + margin_x),
        ]
        if crop.size:
            crops.append(crop)
    return crops


def analyze_video(
    video_path: str | Path,
    checkpoint_path: str | Path,
    frame_interval: int = FRAME_INTERVAL,
    device_name: str | None = None,
):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_name = load_checkpoint(checkpoint_path, device)
    detector = create_scrfd_detector()
    transform = get_eval_transform()
    capture = cv2.VideoCapture(str(video_path))
    frame_scores = []
    frame_index = 0

    with torch.no_grad():
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_interval == 0:
                crops = crop_faces(frame, detector)
                if crops:
                    batch = torch.stack([
                        transform(image=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))["image"]
                        for crop in crops
                    ]).to(device)
                    probabilities = torch.sigmoid(model(batch)).view(-1).cpu().numpy()
                    frame_scores.append(float(np.mean(probabilities)))
            frame_index += 1
    capture.release()

    fake_probability = float(np.mean(frame_scores)) if frame_scores else 0.0
    return {
        "video": str(video_path),
        "model": model_name,
        "frames_with_faces": len(frame_scores),
        "fake_probability": fake_probability,
        "real_probability": 1.0 - fake_probability,
        "prediction": "fake" if fake_probability >= 0.5 else "real",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyser une vidéo avec SCRFD et ResNet-50")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT_DIR / "best_model.pth"))
    parser.add_argument("--frame-interval", type=int, default=FRAME_INTERVAL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(analyze_video(args.video, args.checkpoint, args.frame_interval, args.device))