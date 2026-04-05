"""
Usage:
    python evaluate.py \
        --checkpoint_path '/content/drive/MyDrive/alzheimers_checkpoints/MedViT_small_Dementia.pth' \
        --test_dir '/content/drive/MyDrive/SP26_dementia_data/processed_data/test' \
        --model_name 'MedViT_small'
"""

import argparse
import os
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large

# ── Class names — must match the alphabetical order ImageFolder assigns ──────
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

model_classes = {
    'MedViT_tiny':  MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base':  MedViT_base,
    'MedViT_large': MedViT_large,
}


def load_model(model_name, checkpoint_path, nb_classes, device):
    model_class = model_classes[model_name]
    net = model_class(num_classes=nb_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        saved_epoch = checkpoint.get('epoch', '?')
        saved_acc   = checkpoint.get('acc', '?')
        print(f"Loaded checkpoint from epoch {saved_epoch}, best val acc: {saved_acc:.4f}")
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def build_test_loader(test_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    loader  = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"\nTest set: {len(dataset)} images across {len(dataset.classes)} classes")
    print(f"Class → index mapping: {dataset.class_to_idx}\n")
    return loader, dataset


def evaluate(net, loader, device, nb_classes):
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            outputs = net(images.to(device))
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(all_labels, all_preds, all_probs, nb_classes):
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, digits=4))

    # Per-class metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall    = recall_score(all_labels, all_preds, average='weighted')
    f1        = f1_score(all_labels, all_preds, average='weighted')
    overall   = np.mean(all_labels == all_preds)

    # AUC
    labels_onehot = label_binarize(all_labels, classes=list(range(nb_classes)))
    try:
        auc = roc_auc_score(labels_onehot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')

    # Confusion matrix for specificity
    cm = confusion_matrix(all_labels, all_preds)
    specificities = []
    for i in range(nb_classes):
        tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    print("=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    print(f"  Overall Accuracy : {overall:.4f}")
    print(f"  Weighted F1      : {f1:.4f}")
    print(f"  Weighted Precision: {precision:.4f}")
    print(f"  Weighted Recall  : {recall:.4f}")
    print(f"  Avg Specificity  : {np.mean(specificities):.4f}")
    print(f"  AUC (OvR)        : {auc:.4f}")

    print("\nPer-class specificity:")
    for name, spec in zip(CLASS_NAMES, specificities):
        print(f"  {name:<22}: {spec:.4f}")

    return cm


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=11)

    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_title('Confusion Matrix — Test Set', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nConfusion matrix saved to: {save_path}")


def main(args):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_classes = len(CLASS_NAMES)

    print(f"Device : {device}")
    print(f"Model  : {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Test dir  : {args.test_dir}\n")

    net                         = load_model(args.model_name, args.checkpoint_path, nb_classes, device)
    loader, _                   = build_test_loader(args.test_dir, args.batch_size)
    all_labels, all_preds, all_probs = evaluate(net, loader, device, nb_classes)
    cm                          = print_metrics(all_labels, all_preds, all_probs, nb_classes)

    plot_confusion_matrix(cm, save_path=args.cm_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MedViTV2 on Dementia test split.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to .pth checkpoint.')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test split folder (with class subfolders).')
    parser.add_argument('--model_name', type=str, default='MedViT_small',
                        choices=['MedViT_tiny', 'MedViT_small', 'MedViT_base', 'MedViT_large'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cm_save_path', type=str, default='confusion_matrix.png',
                        help='Where to save confusion matrix image.')
    args = parser.parse_args()
    main(args)