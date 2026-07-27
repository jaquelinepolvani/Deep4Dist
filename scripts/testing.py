"""
Evaluation script for the trained Deep4Dist segmentation model.

Loads best_model.pt (the checkpoint with the lowest validation loss,
NOT the final epoch's weights) and evaluates it on the held-out test
set, computing:
    - overall pixel accuracy
    - per-class IoU and F1
    - a confusion matrix (both raw counts and row-normalized)

Class mapping (confirmed against the dataset):
    0 = background
    1 = bark beetle
    2 = clear-cut
    3 = windthrow
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import CHECKPOINT_DIR
from dataset_provider import get_loader

NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Bark Beetle", "Clear-cut", "Windthrow"]

BATCH_SIZE = 4
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
CONF_MATRIX_PLOT_PATH = CHECKPOINT_DIR / "confusion_matrix.png"


def update_confusion_matrix(conf_mat, preds, targets, num_classes):
    """
    Accumulates a confusion matrix across batches.

    conf_mat[true_class, pred_class] += count of pixels with that
    (true, predicted) pair. Uses np.bincount on a flattened combined
    index instead of a Python loop, since a per-pixel loop over
    hundreds of millions of pixels would be far too slow.
    """
    preds = preds.flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()

    combined_index = targets * num_classes + preds
    counts = np.bincount(combined_index, minlength=num_classes ** 2)
    conf_mat += counts.reshape(num_classes, num_classes)
    return conf_mat


def compute_per_class_metrics(conf_mat):
    """
    Given a (num_classes, num_classes) confusion matrix
    (rows = true class, cols = predicted class), computes per-class
    IoU and F1.
    """
    num_classes = conf_mat.shape[0]
    ious = []
    f1s = []

    for c in range(num_classes):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp   # predicted as c, but wasn't
        fn = conf_mat[c, :].sum() - tp   # was c, but predicted as something else

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

        ious.append(iou)
        f1s.append(f1)

    return ious, f1s


def plot_confusion_matrix(conf_mat, class_names, save_path):
    # Row-normalize: each row sums to 1, showing what fraction of each
    # TRUE class's pixels ended up predicted as each class. This is
    # usually more informative than raw counts on an imbalanced dataset,
    # where raw counts would be dominated by the background class.
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    normalized = np.divide(
        conf_mat, row_sums, out=np.zeros_like(conf_mat, dtype=float), where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Two-line title: short main title + explanatory subtitle, so
    # neither line has to be squeezed to fit and gets cut off.
    ax.set_title("Confusion Matrix\n(row-normalized: recall per true class)", fontsize=12, pad=12)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                     color="white" if normalized[i, j] > 0.5 else "black")

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true-class pixels")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No best_model.pt found at {BEST_MODEL_PATH}. "
            f"Make sure training has run and saved at least one checkpoint."
        )

    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=5, out_channels=4, init_features=32, pretrained=False
    )
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    test_ds = get_loader(dataset_type="test", batch_size=BATCH_SIZE)

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for data, target in tqdm(test_ds, desc="Evaluating on test set"):
            data = data.to(device)
            target = target.squeeze(1).to(device)

            prediction = model(data)
            pred_classes = prediction.argmax(dim=1)

            conf_mat = update_confusion_matrix(conf_mat, pred_classes, target, NUM_CLASSES)

    overall_accuracy = np.trace(conf_mat) / conf_mat.sum()
    ious, f1s = compute_per_class_metrics(conf_mat)

    print("\n=== Test Set Results ===")
    print(f"Overall pixel accuracy: {overall_accuracy:.4f}\n")

    print(f"{'Class':<15} {'IoU':>8} {'F1':>8} {'Pixel count':>15}")
    for name, iou, f1, count in zip(CLASS_NAMES, ious, f1s, conf_mat.sum(axis=1)):
        print(f"{name:<15} {iou:>8.4f} {f1:>8.4f} {count:>15,}")

    print(f"\nMean IoU (macro): {np.nanmean(ious):.4f}")
    print(f"Mean F1 (macro):  {np.nanmean(f1s):.4f}")

    print(f"\nConfusion matrix (raw counts, rows=true, cols=predicted):\n{conf_mat}")

    plot_confusion_matrix(conf_mat, CLASS_NAMES, CONF_MATRIX_PLOT_PATH)