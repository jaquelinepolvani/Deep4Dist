"""
Compares predictions from all 4 trained models across 3 different
"Mixed disturbances" test samples (the top-3 candidates by how
balanced their mix of bark beetle / clear-cut / windthrow pixels is).

Layout: 3 rows (one per mixed sample) x 6 columns
(RGB | Ground Truth | Run 1 | Run 2 | Run 3 | Run 4)

Sample selection is based ONLY on ground truth pixel counts, never on
any model's prediction, so all 4 models are compared on identical,
fairly-chosen inputs.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

from config import CHECKPOINT_DIR
from dataset_provider import Deep4DistDataset

NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Bark Beetle", "Clear-cut", "Windthrow"]

CLASS_COLORS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.65, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

RUNS = [
    {"label": "Run 1\n(CE only)", "checkpoint_dir": CHECKPOINT_DIR.parent / "checkpoints_1sttry"},
    {"label": "Run 2\n(CE+Dice, w=0.5)", "checkpoint_dir": CHECKPOINT_DIR.parent / "checkpoints_2ndtry"},
    {"label": "Run 3\n(CE+Dice, w=0.4, +Aug)", "checkpoint_dir": CHECKPOINT_DIR.parent / "checkpoints_3rdtry"},
    {"label": "Run 4\n(CE+Dice, w=0.5, +Aug)", "checkpoint_dir": CHECKPOINT_DIR.parent / "checkpoints_4thtry"},
]

NUM_MIXED_SAMPLES = 3  # how many different "Mixed" examples to show as rows

OUTPUT_PATH = CHECKPOINT_DIR.parent / "figures" / "model_comparison_3mixed_samples.png"


def mask_to_rgb(mask):
    return CLASS_COLORS[mask]


def find_mixed_sample_indices(dataset, top_k):
    """
    Finds the top_k samples with the most balanced mix of all 3
    disturbance classes at once (smallest spread between their pixel
    counts). Deterministic — depends only on ground truth, so it's
    identical regardless of which model is used later.
    """
    class_pixel_counts = np.zeros((len(dataset), NUM_CLASSES), dtype=np.int64)

    for i in tqdm(range(len(dataset)), desc="Scanning test set for mixed samples"):
        _, mask = dataset[i]
        counts = np.bincount(mask.flatten().numpy(), minlength=NUM_CLASSES)
        class_pixel_counts[i] = counts

    disturbance_counts = class_pixel_counts[:, 1:]
    has_all_three = np.all(disturbance_counts > 0, axis=1)
    spread = disturbance_counts.max(axis=1) - disturbance_counts.min(axis=1)
    spread_for_ranking = np.where(has_all_three, spread, np.iinfo(np.int64).max)

    top_k_indices = np.argsort(spread_for_ranking)[:top_k]
    return top_k_indices.tolist()


def load_model(checkpoint_dir, device):
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=5, out_channels=4, init_features=32, pretrained=False
    )
    state_dict = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    test_dataset = Deep4DistDataset(dataset_type="test")

    mixed_indices = find_mixed_sample_indices(test_dataset, top_k=NUM_MIXED_SAMPLES)
    print(f"Selected 'Mixed' sample indices: {mixed_indices}")

    # Pre-load each sample's image/mask once, since it's reused across all 4 models.
    samples = []
    for idx in mixed_indices:
        image, true_mask = test_dataset[idx]
        samples.append({
            "idx": idx,
            "image": image,
            "true_mask_np": true_mask.squeeze(0).numpy(),
            "rgb_image": image[:3].permute(1, 2, 0).numpy(),
        })

    num_rows = len(samples)
    num_cols = 2 + len(RUNS)  # RGB + Ground Truth + one per run

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.3 * num_cols, 3.3 * num_rows))

    # Column headers (only on the top row)
    col_titles = ["RGB Input", "Ground Truth"] + [run["label"] for run in RUNS]

    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["rgb_image"])
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_to_rgb(sample["true_mask_np"]))
        axes[row, 1].axis("off")

        for i, run in enumerate(RUNS):
            print(f"Row {row + 1}/{num_rows}: predicting with {run['label'].splitlines()[0]}...")
            model = load_model(run["checkpoint_dir"], device)

            with torch.no_grad():
                pred = model(sample["image"].unsqueeze(0).to(device))
                pred_mask = pred.argmax(dim=1).squeeze(0).cpu().numpy()

            ax = axes[row, 2 + i]
            ax.imshow(mask_to_rgb(pred_mask))
            ax.axis("off")

        # Row label on the far left, showing which sample index this is
        axes[row, 0].set_ylabel(f"Sample idx {sample['idx']}", fontsize=10)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

    # Column titles only on the top row
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)

    legend_elements = [
        Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle("Model Comparison Across 3 'Mixed Disturbances' Test Samples", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
    plt.show()

    print(f"\nSaved to {OUTPUT_PATH}")