"""
Usage in Colab:
    !pip install autogluon.multimodal -q

    !python autogluon_train.py \
        --train_dir '/content/drive/MyDrive/SP26_dementia_data/processed_data/train' \
        --val_dir   '/content/drive/MyDrive/SP26_dementia_data/processed_data/val' \
        --test_dir  '/content/drive/MyDrive/SP26_dementia_data/processed_data/test' \
        --save_dir  '/content/drive/MyDrive/alzheimers_autogluon' \
        --time_limit 3600
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize


CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']


def build_dataframe(split_dir):
    rows = []
    split_dir = Path(split_dir)
    for class_folder in sorted(split_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        label = class_folder.name
        for img_path in class_folder.glob('*'):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                rows.append({'image': str(img_path), 'label': label})
    df = pd.DataFrame(rows)
    print(f"  {split_dir.name}: {len(df)} images, "
          f"classes: {df['label'].value_counts().to_dict()}")
    return df


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_title('Confusion Matrix — AutoGluon (Test Set)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to: {save_path}")


def print_metrics(all_labels, all_preds, all_probs):
    nb_classes = len(CLASSES)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds,
                                target_names=CLASSES, digits=4))

    overall = np.mean(np.array(all_labels) == np.array(all_preds))
    f1      = f1_score(all_labels, all_preds, average='weighted')

    labels_onehot = label_binarize(
        [CLASSES.index(l) for l in all_labels],
        classes=list(range(nb_classes))
    )
    try:
        auc = roc_auc_score(labels_onehot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(all_labels, all_preds, labels=CLASSES)
    specificities = []
    for i in range(nb_classes):
        tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    print("=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    print(f"  Overall Accuracy  : {overall:.4f}")
    print(f"  Weighted F1       : {f1:.4f}")
    print(f"  AUC (OvR)         : {auc:.4f}")
    print(f"  Avg Specificity   : {np.mean(specificities):.4f}")
    print("\nPer-class specificity:")
    for name, spec in zip(CLASSES, specificities):
        print(f"  {name:<22}: {spec:.4f}")

    return cm


def main(args):
    from autogluon.multimodal import MultiModalPredictor

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, 'ag_model')

    print("\nBuilding datasets...")
    train_df = build_dataframe(args.train_dir)
    val_df   = build_dataframe(args.val_dir)
    test_df  = build_dataframe(args.test_dir)

    # Combine train + val so AutoGluon can do its own internal tuning,
    # or keep them separate — here we pass val as tuning data explicitly
    print(f"\nTotal train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    if os.path.exists(model_path) and not args.retrain:
        print(f"\nFound existing model at {model_path}, loading it.")
        print("Pass --retrain to train from scratch.")
        predictor = MultiModalPredictor.load(model_path)
    else:
        print(f"\nTraining AutoGluon model (time limit: {args.time_limit}s)...")
        print("AutoGluon will try multiple CNN backbones and pick the best.\n")

        predictor = MultiModalPredictor(
            label='label',
            problem_type='multiclass',
            eval_metric='acc',
            path=model_path,
        )

        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,          
            time_limit=args.time_limit,
            hyperparameters={
                'model.timm_image.checkpoint_name': args.backbone,
            },
        )

        print("\nTraining complete.")
        predictor.save(model_path)
        print(f"Model saved to: {model_path}")

    print("\nRunning inference on test set...")
    test_probs_df = predictor.predict_proba(test_df)
    test_preds    = predictor.predict(test_df).tolist()
    test_labels   = test_df['label'].tolist()

    all_probs = test_probs_df[CLASSES].values

    cm = print_metrics(test_labels, test_preds, all_probs)

    cm_path = os.path.join(args.save_dir, 'autogluon_confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)

    print("\n" + "=" * 60)
    print("AUTOGLUON LEADERBOARD (val performance)")
    print("=" * 60)
    try:
        lb = predictor.leaderboard(val_df, silent=True)
        print(lb[['model', 'score_val', 'pred_time_val']].to_string(index=False))
    except Exception:
        pass  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AutoGluon CNN baseline for Alzheimer\'s MRI classification.')
    parser.add_argument('--train_dir',   type=str, required=True)
    parser.add_argument('--val_dir',     type=str, required=True)
    parser.add_argument('--test_dir',    type=str, required=True)
    parser.add_argument('--save_dir',    type=str, required=True,
                        help='Where to save the model and outputs.')
    parser.add_argument('--time_limit',  type=int, default=3600,
                        help='Training time limit in seconds (default: 1 hour).')
    parser.add_argument('--backbone',    type=str, default='resnet50',
                        help='timm backbone for AutoGluon to use. '
                             'Examples: resnet50, efficientnet_b3, convnext_tiny')
    parser.add_argument('--retrain',     action='store_true',
                        help='Force retrain even if a saved model exists.')
    args = parser.parse_args()
    main(args)