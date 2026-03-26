import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_kfold_loaders(data_root: str, fold_idx: int, k: int = 5, batch_size: int = 32, num_workers: int = 2):
    """
    Splits the training data into K folds and returns loaders for a specific fold.
    
    Args:
        data_root: Path to the root folder (containing a 'train' subfolder)
        fold_idx:  Which fold to use as the validation set (0 to k-1)
        k:         Total number of folds (default: 5)
    """

    # 1. Standard Transform (No Augmentation)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 2. Load the full training dataset
    # Point this to your 'train' folder which now acts as the total pool
    full_dataset = ImageFolder(root=f"{data_root}/train", transform=transform)
    
    # 3. Define the K-Fold Split
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = list(range(len(full_dataset)))
    
    # Get the indices for the requested fold
    train_indices, val_indices = list(kf.split(indices))[fold_idx]

    # 4. Create Subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset   = Subset(full_dataset, val_indices)

    # 5. Wrap in DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Separate Test Loader (stays constant)
    test_dataset = ImageFolder(root=f"{data_root}/test", transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_dataset.class_to_idx

if __name__ == "__main__":
    DATA_ROOT = "/content/drive/MyDrive/Software SP26 Alzheimer's Detection/processed_data"
    
    # Example: Load Fold 0
    train_loader, val_loader, test_loader, class_to_idx = get_kfold_loaders(DATA_ROOT, fold_idx=0)
    
    print(f"Fold 0: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples")