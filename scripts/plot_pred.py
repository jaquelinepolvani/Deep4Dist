"""
For each disturbance category, selects the top-5 test samples ranked
by ground-truth pixel count, then plots the RGB input, ground truth,
and the model's prediction for a chosen candidate.

Sample selection is based only on ground truth, never on the model's
predictions, keeping results comparable across different models.

RANK selects which candidate to use per category (1 = highest
ground-truth pixel count, 2 = second-highest, and so on).
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

# Which candidate to pick per category: 1 = most dominant (used before),
# 2 = second-most dominant, etc. Try a few values until the selected
# images clearly show the pattern you want to illustrate.
RANK = 2

# Path to the checkpoint directory of the model to visualize
# (e.g. checkpoints_2ndtry for Run 2).
MODEL_CHECKPOINT_DIR = CHECKPOINT_DIR.parent / "checkpoints_3rdtry"

BEST_MODEL_PATH = MODEL_CHECKPOINT_DIR / "best_model.pt"
OUTPUT_PATH = MODEL_CHECKPOINT_DIR / "prediction_examples_v2.png"


def mask_to_rgb(mask):
    return CLASS_COLORS[mask]


def find_candidate_samples(dataset, rank=1, top_k=5):
    """
    Scans the whole test set and, for each disturbance class plus a
    "mixed" category, returns the index ranked `rank` (1 = highest)
    among the top_k candidates by ground-truth pixel count.

    """
    class_pixel_counts = np.zeros((len(dataset), NUM_CLASSES), dtype=np.int64)

    for i in tqdm(range(len(dataset)), desc="Scanning test set"):
        _, mask = dataset[i]
        counts = np.bincount(mask.flatten().numpy(), minlength=NUM_CLASSES)
        class_pixel_counts[i] = counts

    results = {}

    # Per-class dominant examples (rank-th highest pixel count for that class)
    for class_idx, class_name in [(1, "Bark Beetle"), (2, "Clear-cut"), (3, "Windthrow")]:
        counts_for_class = class_pixel_counts[:, class_idx]
        top_k_indices = np.argsort(counts_for_class)[::-1][:top_k]
        chosen = int(top_k_indices[min(rank - 1, len(top_k_indices) - 1)])
        results[class_name] = (chosen, top_k_indices.tolist())

    # "Mixed" category: samples containing all 3 disturbance types,
    # ranked by how balanced the mix is (smallest spread = most balanced)
    disturbance_counts = class_pixel_counts[:, 1:]
    has_all_three = np.all(disturbance_counts > 0, axis=1)
    spread = disturbance_counts.max(axis=1) - disturbance_counts.min(axis=1)
    spread_for_ranking = np.where(has_all_three, spread, np.iinfo(np.int64).max)
    top_k_mixed = np.argsort(spread_for_ranking)[:top_k]
    chosen_mixed = int(top_k_mixed[min(rank - 1, len(top_k_mixed) - 1)])
    results["Mixed"] = (chosen_mixed, top_k_mixed.tolist())

    return results


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No best_model.pt found at {BEST_MODEL_PATH}.")

    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=5, out_channels=4, init_features=32, pretrained=False
    )
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    test_dataset = Deep4DistDataset(dataset_type="test")

    candidates = find_candidate_samples(test_dataset, rank=RANK, top_k=5)

    print(f"\nUsing rank={RANK} for each category:")
    for name, (chosen_idx, top_k_list) in candidates.items():
        print(f"  {name}: chosen index={chosen_idx} (top-{len(top_k_list)} candidates were: {top_k_list})")

    samples = {name: idx for name, (idx, _) in candidates.items()}
    num_examples = len(samples)

    fig, axes = plt.subplots(3, num_examples, figsize=(3.3 * num_examples, 10))
    row_labels = ["RGB Input", "Ground Truth", "Prediction"]

    for col, (title, idx) in enumerate(samples.items()):
        image, true_mask = test_dataset[idx]

        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred_mask = pred.argmax(dim=1).squeeze(0).cpu().numpy()

        true_mask_np = true_mask.squeeze(0).numpy()
        rgb_image = image[:3].permute(1, 2, 0).numpy()

        axes[0, col].imshow(rgb_image)
        axes[0, col].set_title(f"{title}\n(rank {RANK}, idx {idx})", fontsize=11)
        axes[0, col].axis("off")

        axes[1, col].imshow(mask_to_rgb(true_mask_np))
        axes[1, col].axis("off")

        axes[2, col].imshow(mask_to_rgb(pred_mask))
        axes[2, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(label, fontsize=12)

    legend_elements = [
        Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
    plt.show()

    print(f"\nSaved to {OUTPUT_PATH}")
