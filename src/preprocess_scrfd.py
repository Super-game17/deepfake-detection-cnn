import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
from insightface.app import FaceAnalysis

from preprocess import collect_celeb_entries, collect_ff_entries, assign_splits


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed_scrfd"
IMAGE_SIZE = (224, 224)
FRAME_INTERVAL = 15
MIN_FACE_SIZE = 60
MIN_CONFIDENCE = 0.50


def create_detector():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA est obligatoire pour le preprocessing SCRFD.")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    detector = FaceAnalysis(name="buffalo_l", providers=providers)
    detector.prepare(ctx_id=0, det_size=(640, 640))
    return detector


def extract_faces_scrfd(video_path, output_folder, prefix, detector):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[AVERTISSEMENT] Vidéo impossible à ouvrir: {video_path}")
        return []

    saved = []
    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_count % FRAME_INTERVAL == 0:
            faces = detector.get(frame)
            for face_index, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox.astype(int)
                width = x2 - x1
                height = y2 - y1
                if face.det_score < MIN_CONFIDENCE or min(width, height) < MIN_FACE_SIZE:
                    continue
                margin_x = int(0.1 * width)
                margin_y = int(0.1 * height)
                crop = frame[
                    max(0, y1 - margin_y):min(frame.shape[0], y2 + margin_y),
                    max(0, x1 - margin_x):min(frame.shape[1], x2 + margin_x),
                ]
                if crop.size == 0:
                    continue
                output_path = output_folder / f"{prefix}_f{frame_count}_face{face_index}.jpg"
                resized = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                if cv2.imwrite(str(output_path), resized):
                    saved.append(output_path)
        frame_count += 1
    capture.release()
    return saved


def process_dataset(extract=True):
    entries = collect_celeb_entries() + collect_ff_entries()
    assign_splits(entries)
    detector = create_detector() if extract else None
    rows = []

    for index, entry in enumerate(entries, start=1):
        output_folder = OUTPUT_DIR / entry["split"] / entry["label"] / entry["dataset"]
        output_folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{entry['dataset']}_{entry['label']}_{entry['path'].stem}"
        images = (
            extract_faces_scrfd(entry["path"], output_folder, prefix, detector)
            if extract
            else []
        )
        if extract and not images:
            print(f"[AVERTISSEMENT] Aucun visage SCRFD: {entry['path']}")
        for image_path in images:
            rows.append({
                "image_path": str(image_path.relative_to(BASE_DIR)),
                "dataset": entry["dataset"],
                "label": entry["label"],
                "label_binary": int(entry["label"] == "fake"),
                "split": entry["split"],
                "source_video": str(entry["path"].relative_to(RAW_DIR)),
                "source_group": entry["source_group"],
                "manipulation_method": entry["manipulation_method"],
                "detector": "scrfd",
            })
        if index % 100 == 0:
            print(f"Vidéos traitées: {index}/{len(entries)}")

    dataframe = pd.DataFrame(rows)
    if extract:
        csv_path = OUTPUT_DIR / "metadata_scrfd.csv"
        dataframe.to_csv(csv_path, index=False)
        print(f"Métadonnées SCRFD enregistrées sous: {csv_path}")
        if not dataframe.empty:
            print(dataframe.groupby(["dataset", "split", "label"]).size())
    else:
        print(f"Vidéos valides trouvées: {len(entries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing SCRFD sur GPU")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    process_dataset(extract=not args.dry_run)