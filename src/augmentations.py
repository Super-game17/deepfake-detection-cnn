"""Transformations appliquees pendant l'entrainement.

Les images de validation et de test ne recoivent aucune augmentation aleatoire.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform():
    """Retourne les augmentations utilisees uniquement pour le train."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.35,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.ImageCompression(quality_range=(60, 95)),
        ], p=0.25),
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform():
    """Retourne la transformation deterministe pour validation et test."""
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])