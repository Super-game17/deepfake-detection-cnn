import argparse
import re
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CELEB_DIR = RAW_DIR / "Celeb-DF-v2"
FF_DIR = RAW_DIR / "FaceForensics++_C23"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CELEB_TEST_LIST = CELEB_DIR / "List_of_testing_videos.txt"
FF_METADATA = FF_DIR / "csv" / "FF++_Metadata.csv"
FRAME_INTERVAL = 15
IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def normalize_path(value):
    return str(value).replace("\\", "/").lstrip("./")


def load_celeb_test_paths():
    if not CELEB_TEST_LIST.exists():
        raise FileNotFoundError(f"Liste Celeb-DF introuvable: {CELEB_TEST_LIST}")
    paths = set()
    for line in CELEB_TEST_LIST.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            paths.add(normalize_path(parts[1]))
    return paths


def add_entry(entries, path, dataset, label, group, split, method):
    if path.exists():
        entries.append({
            "path": path,
            "dataset": dataset,
            "label": label,
            "source_group": group,
            "split": split,
            "manipulation_method": method,
        })


def collect_celeb_entries():
    test_paths = load_celeb_test_paths()
    entries = []
    folders = {
        "Celeb-real": ("real", "real"),
        "YouTube-real": ("real", "real"),
        "Celeb-synthesis": ("fake", "Celeb-synthesis"),
    }
    for folder, (label, method) in folders.items():
        for path in sorted((CELEB_DIR / folder).glob("*.mp4")):
            relative = normalize_path(path.relative_to(CELEB_DIR))
            group = f"celeb_{path.stem.split('_')[0]}"
            split = "test" if relative in test_paths else "pending"
            add_entry(entries, path, "celeb_df", label, group, split, method)
    return entries


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def collect_ff_entries():
    if not FF_METADATA.exists():
        raise FileNotFoundError(f"Métadonnées FaceForensics++ introuvables: {FF_METADATA}")
    metadata = pd.read_csv(FF_METADATA, usecols=["File Path", "Label"])
    paths = [FF_DIR / normalize_path(value) for value in metadata["File Path"]]
    union_find = UnionFind()
    pair_pattern = re.compile(r"^(\d+)_(\d+)$")
    for path in paths:
        match = pair_pattern.match(path.stem)
        if match:
            union_find.union(match.group(1), match.group(2))

    entries = []
    for row, path in zip(metadata.itertuples(index=False), paths):
        relative = Path(normalize_path(row[0]))
        method = relative.parts[0]
        label = "real" if str(row[1]).upper() == "REAL" else "fake"
        match = pair_pattern.match(path.stem)
        if match:
            group = f"ff_{union_find.find(match.group(1))}"
        else:
            group = f"ff_{path.stem}"
        add_entry(entries, path, "faceforensics", label, group, "pending", method)
    return entries


def assign_splits(entries):
    pending = [entry for entry in entries if entry["split"] == "pending"]
    groups = sorted({entry["source_group"] for entry in pending})
    train_groups, holdout_groups = train_test_split(
        groups, test_size=0.2, random_state=RANDOM_STATE
    )
    val_groups, test_groups = train_test_split(
        holdout_groups, test_size=0.5, random_state=RANDOM_STATE
    )
    for entry in pending:
        if entry["source_group"] in train_groups:
            entry["split"] = "train"
        elif entry["source_group"] in val_groups:
            entry["split"] = "val"
        elif entry["source_group"] in test_groups:
            entry["split"] = "test"
        else:
            raise RuntimeError(f"Groupe non assigné: {entry['source_group']}")


def extract_faces(video_path, output_folder, prefix):
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for face_index, (x, y, width, height) in enumerate(faces):
                margin = int(0.1 * width)
                face = frame[
                    max(0, y - margin):min(frame.shape[0], y + height + margin),
                    max(0, x - margin):min(frame.shape[1], x + width + margin),
                ]
                if face.size == 0:
                    continue
                output_path = output_folder / f"{prefix}_f{frame_count}_face{face_index}.jpg"
                resized = cv2.resize(face, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                if cv2.imwrite(str(output_path), resized):
                    saved.append(output_path)
        frame_count += 1
    capture.release()
    return saved


def build_metadata(entries, extract):
    rows = []
    for index, entry in enumerate(entries, start=1):
        output_folder = PROCESSED_DIR / entry["split"] / entry["label"] / entry["dataset"]
        output_folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{entry['dataset']}_{entry['label']}_{entry['path'].stem}"
        images = extract_faces(entry["path"], output_folder, prefix) if extract else []
        if extract and not images:
            print(f"[AVERTISSEMENT] Aucun visage extrait: {entry['path']}")
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
            })
        if index % 100 == 0:
            print(f"Vidéos traitées: {index}/{len(entries)}")
    return pd.DataFrame(rows)


def print_statistics(entries, dataframe):
    videos = pd.DataFrame(entries)
    print("\nVidéos par dataset, split et classe:")
    print(videos.groupby(["dataset", "split", "label"]).size())
    if not dataframe.empty:
        print("\nImages extraites par dataset, split et classe:")
        print(dataframe.groupby(["dataset", "split", "label"]).size())


def process_dataset(extract=True):
    entries = collect_celeb_entries() + collect_ff_entries()
    assign_splits(entries)
    print(f"Vidéos valides trouvées: {len(entries)}")
    dataframe = build_metadata(entries, extract)
    if extract:
        csv_path = PROCESSED_DIR / "metadata.csv"
        dataframe.to_csv(csv_path, index=False)
        print(f"Métadonnées enregistrées sous: {csv_path}")
    print_statistics(entries, dataframe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    process_dataset(extract=not args.dry_run)