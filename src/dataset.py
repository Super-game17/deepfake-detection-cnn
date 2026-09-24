from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import get_eval_transform, get_train_transform


BASE_DIR = Path(__file__).resolve().parent.parent


class DeepfakeDataset(Dataset):
    """Dataset d'images extraites a partir de metadata.csv."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = BASE_DIR / Path(row["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image impossible a lire: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = torch.tensor(row["label_binary"], dtype=torch.float32)
        return image, label


def load_datasets(metadata_path=None):
    """Construit les datasets train, val et test avec les transforms adaptees."""
    metadata_path = metadata_path or BASE_DIR / "data" / "processed" / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    train = DeepfakeDataset(
        metadata[metadata["split"] == "train"], get_train_transform()
    )
    val = DeepfakeDataset(
        metadata[metadata["split"] == "val"], get_eval_transform()
    )
    test = DeepfakeDataset(
        metadata[metadata["split"] == "test"], get_eval_transform()
    )
    return train, val, test


def create_dataloaders(metadata_path=None, batch_size=32, num_workers=0):
    """Construit les DataLoaders prets pour l'entrainement et l'evaluation."""
    train, val, test = load_datasets(metadata_path)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train, shuffle=True, **loader_kwargs),
        DataLoader(val, shuffle=False, **loader_kwargs),
        DataLoader(test, shuffle=False, **loader_kwargs),
    )