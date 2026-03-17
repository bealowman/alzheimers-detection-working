"""
dataloader.py
=============
DataLoader for the Alzheimer's MRI dataset (processed by pipeline).
Use this in the training script to load train, val, and test data.

Usage:
    from dataloader import get_dataloaders

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_root="/content/drive/MyDrive/Software SP26 Alzheimer's Detection/processed_data",
        batch_size=32
    )

Classes (label → index):
    MildDemented     → 0
    ModerateDemented → 1
    NonDemented      → 2
    VeryMildDemented → 3
"""

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ImageNet normalization constants (required by MedViT V2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 2):
    """
    Returns train, val, and test DataLoaders for the Alzheimer's MRI dataset.

    Args:
        data_root:    Path to processed_data folder (contains train/, val/, test/)
        batch_size:   Number of images per batch (default: 32)
        num_workers:  Parallel workers for loading (default: 2, use 0 on Windows)

    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """

    # Training transforms — includes augmentation to help the model generalize
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Val/test transforms — no augmentation, just resize and normalize
    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Load datasets from processed_data folder structure
    train_dataset = ImageFolder(root=f"{data_root}/train", transform=train_transform)
    val_dataset   = ImageFolder(root=f"{data_root}/val",   transform=eval_transform)
    test_dataset  = ImageFolder(root=f"{data_root}/test",  transform=eval_transform)

    # Wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Classes: {train_dataset.classes}")
    print(f"Class → index: {train_dataset.class_to_idx}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


if __name__ == "__main__":
    # Quick test — run this file directly to verify everything loads correctly
    DATA_ROOT = "/content/drive/MyDrive/Software SP26 Alzheimer's Detection/processed_data"

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(DATA_ROOT)

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")   # should be torch.Size([32, 3, 224, 224])
    print(f"Label sample: {labels}")
