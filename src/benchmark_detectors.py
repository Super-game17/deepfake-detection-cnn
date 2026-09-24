import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "detector_benchmark"
IMAGE_SIZE = (224, 224)


def collect_videos(limit_per_class):
    folders = [
        (RAW_DIR / "Celeb-DF-v2" / "Celeb-real", "celeb_df", "real"),
        (RAW_DIR / "Celeb-DF-v2" / "Celeb-synthesis", "celeb_df", "fake"),
        (RAW_DIR / "FaceForensics++_C23" / "original", "faceforensics", "real"),
        (RAW_DIR / "FaceForensics++_C23" / "Deepfakes", "faceforensics", "fake"),
    ]
    videos = []
    for folder, dataset, label in folders:
        videos.extend(
            (path, dataset, label)
            for path in sorted(folder.glob("*.mp4"))[:limit_per_class]
        )
    return videos


def sample_frames(video_path, frame_interval, max_frames):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    frame_index = 0
    while len(frames) < max_frames:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % frame_interval == 0:
            frames.append((frame_index, frame))
        frame_index += 1
    capture.release()
    return frames


def haar_detector():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [(x, y, x + width, y + height, 1.0) for x, y, width, height in boxes]

    return detect


def mtcnn_detector(device):
    import torch
    from facenet_pytorch import MTCNN

    detector = MTCNN(keep_all=True, device=device, post_process=False)

    def detect(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probabilities = detector.detect(rgb)
        if boxes is None:
            return []
        return [
            (*box.astype(int), float(probability))
            for box, probability in zip(boxes, probabilities)
            if probability is not None and probability >= 0.90
        ]

    return detect


def scrfd_detector():
    from insightface.app import FaceAnalysis

    import torch

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    context_id = 0 if torch.cuda.is_available() else -1
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=context_id, det_size=(640, 640))

    def detect(frame):
        faces = app.get(frame)
        return [
            (*face.bbox.astype(int), float(face.det_score))
            for face in faces
            if face.det_score >= 0.50
        ]

    return detect


def make_detector(name, device):
    if name == "haar":
        return haar_detector()
    if name == "mtcnn":
        return mtcnn_detector(device)
    if name == "scrfd":
        return scrfd_detector()
    raise ValueError(f"Detecteur inconnu: {name}")


def draw_detections(frame, detections):
    annotated = frame.copy()
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            annotated,
            f"{score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_benchmark(detector_name, videos, frame_interval, max_frames, device):
    detector = make_detector(detector_name, device)
    rows = []
    annotated_dir = OUTPUT_DIR / detector_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    for video_path, dataset, label in videos:
        frames = sample_frames(video_path, frame_interval, max_frames)
        video_detections = 0
        empty_frames = 0
        for frame_index, frame in frames:
            detections = detector(frame)
            video_detections += len(detections)
            empty_frames += int(not detections)
            if len(rows) < 12:
                annotated = draw_detections(frame, detections)
                output_path = annotated_dir / f"{dataset}_{label}_{video_path.stem}_{frame_index}.jpg"
                cv2.imwrite(str(output_path), annotated)
        rows.append({
            "detector": detector_name,
            "video": str(video_path.relative_to(RAW_DIR)),
            "dataset": dataset,
            "label": label,
            "frames": len(frames),
            "detections": video_detections,
            "empty_frames": empty_frames,
        })

    elapsed = time.perf_counter() - start
    return rows, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark de detecteurs de visages")
    parser.add_argument("--detectors", nargs="+", default=["haar", "mtcnn", "scrfd"])
    parser.add_argument("--limit-per-class", type=int, default=10)
    parser.add_argument("--frame-interval", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    videos = collect_videos(args.limit_per_class)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    print(f"Vidéos benchmarkées: {len(videos)} | device: {device}")

    for detector_name in args.detectors:
        try:
            rows, elapsed = run_benchmark(
                detector_name, videos, args.frame_interval, args.max_frames, device
            )
        except Exception as error:
            print(f"[IGNORÉ] {detector_name}: {error}")
            continue
        all_rows.extend(rows)
        total_frames = sum(row["frames"] for row in rows)
        total_detections = sum(row["detections"] for row in rows)
        empty_frames = sum(row["empty_frames"] for row in rows)
        summary.append({
            "detector": detector_name,
            "elapsed_seconds": round(elapsed, 2),
            "frames": total_frames,
            "detections": total_detections,
            "empty_frames": empty_frames,
            "detections_per_frame": round(total_detections / max(1, total_frames), 3),
            "empty_rate": round(empty_frames / max(1, total_frames), 3),
        })

    with (OUTPUT_DIR / "details.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0].keys()) if all_rows else None
        if writer:
            writer.writeheader()
            writer.writerows(all_rows)
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys()) if summary else None
        if writer:
            writer.writeheader()
            writer.writerows(summary)
    print("\nRésultats:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()