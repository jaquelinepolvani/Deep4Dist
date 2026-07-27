"""
Training script for a segmentation model on the Deep4Dist dataset.

Model: U-Net (mateuszbuda/brain-segmentation-pytorch), trained from
scratch (pretrained=False), adapted to:
    - 5 input channels (RGB + NIR + normalized DSM)
    - 4 output classes (0=background, 1=bark beetle, 2=clear-cut, 3=windthrow)

This version adds:
    - Data augmentation (random 90-degree rotations + horizontal/
      vertical flips), applied only during training, not validation
    - Combined CrossEntropy + Dice loss with dice_weight=0.4

Other features (unchanged from before):
    - GPU support (falls back to CPU with a warning if no GPU found)
    - Checkpointing after every epoch (last_checkpoint.pt)
    - Automatic resume from checkpoint
    - Best-model tracking (best_model.pt), based on validation loss
    - Early stopping
    - LR scheduler on validation plateau
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from tqdm import tqdm

from config import BASE_PATH, CHECKPOINT_DIR
from dataset_provider import get_loader

# --- Reproducibility ---------------------------------------------------
SEED = 42

# --- Fixed hyperparameters ----------------------------------------------
BATCH_SIZE = 4
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 7
LEARNING_RATE = 0.001
NUM_CLASSES = 4

# Real class weights, computed via median-frequency-balancing on the
# actual train-set pixel counts (background, bark beetle, clear-cut,
# windthrow).
CLASS_WEIGHTS = [0.229, 1.345, 0.796, 16.479]

LOSS_TYPE = "combined"
DICE_WEIGHT = 0.5

LAST_CKPT_PATH = CHECKPOINT_DIR / "last_checkpoint.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
CURVES_PATH = CHECKPOINT_DIR / "training_curves.png"


# =========================================================================
# Data augmentation
# =========================================================================

def augment_batch(images, masks):
    """
    Applies random 90-degree rotation + horizontal/vertical flip to
    each sample in the batch. Image and mask always get the EXACT
    SAME transform per sample, so they stay pixel-aligned.

    Applied independently per sample (not once for the whole batch),
    so a batch of 4 images can get 4 different augmentations, giving
    more variety per training step.

    images: (B, C, H, W) float tensor
    masks:  (B, H, W) long tensor (class indices)
    """
    batch_size = images.shape[0]

    for i in range(batch_size):
        # Random 90-degree rotation: 0, 1, 2, or 3 times 90 degrees.
        # torch.rot90 rotates over the last two dims (H, W), which
        # works the same way for a (C, H, W) image and a (H, W) mask.
        k = random.randint(0, 3)
        if k > 0:
            images[i] = torch.rot90(images[i], k=k, dims=(1, 2))
            masks[i] = torch.rot90(masks[i], k=k, dims=(0, 1))

        # Random horizontal flip
        if random.random() < 0.5:
            images[i] = torch.flip(images[i], dims=(2,))
            masks[i] = torch.flip(masks[i], dims=(1,))

        # Random vertical flip
        if random.random() < 0.5:
            images[i] = torch.flip(images[i], dims=(1,))
            masks[i] = torch.flip(masks[i], dims=(0,))

    return images, masks


# =========================================================================
# Loss functions
# =========================================================================

class DiceLoss(nn.Module):
    """
    Measures region overlap (similar to IoU) between predicted and true
    class regions, rather than per-pixel probability like CrossEntropy.
    This gives a much stronger training signal for rare classes, since
    it isn't diluted by a huge number of easy, correctly-classified
    background pixels the way a per-pixel average loss is.
    """

    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """Weighted CrossEntropy + Dice, blended by dice_weight."""

    def __init__(self, class_weights, num_classes, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (1 - self.dice_weight) * ce + self.dice_weight * dice


def build_loss_fn(loss_type, class_weights, num_classes, dice_weight):
    if loss_type == "crossentropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "dice":
        return DiceLoss(num_classes=num_classes)
    elif loss_type == "combined":
        return CombinedLoss(class_weights=class_weights, num_classes=num_classes, dice_weight=dice_weight)
    else:
        raise ValueError(f"Unknown LOSS_TYPE '{loss_type}'.")


# =========================================================================
# Train / validation loops
# =========================================================================

def train(model, loss_fn, optimizer, epoch, train_ds, device):
    model.train()

    running_loss = []
    running_acc = []

    loop = tqdm(train_ds)

    for (data, target) in loop:
        data = data.to(device)
        target = target.squeeze(1).to(device)

        # Augmentation only during training, never during validation —
        # validation should always measure performance on the real,
        # unmodified data.
        data, target = augment_batch(data, target)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(data)

        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

        pred_classes = prediction.argmax(dim=1)
        acc = (pred_classes == target).float().mean().item()

        running_loss.append(loss.item())
        running_acc.append(acc)
        loop.set_postfix_str(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc:.4f}")

    return np.mean(running_loss), np.mean(running_acc)


def validation(model, loss_fn, epoch, valid_ds, device):
    model.eval()

    running_loss = []
    running_acc = []

    loop = tqdm(valid_ds)

    with torch.no_grad():
        for (data, target) in loop:
            data = data.to(device)
            target = target.squeeze(1).to(device)

            # No augmentation here — validation measures real performance.

            prediction = model(data)
            loss = loss_fn(prediction, target)

            pred_classes = prediction.argmax(dim=1)
            acc = (pred_classes == target).float().mean().item()

            running_loss.append(loss.item())
            running_acc.append(acc)
            loop.set_postfix_str(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc:.4f}")

    return np.mean(running_loss), np.mean(running_acc)


if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cpu":
        print("WARNING: No GPU found — training on CPU will be substantially slower.")

    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=5, out_channels=4, init_features=32, pretrained=False
    )
    model = model.to(device)

    class_weights = torch.tensor(CLASS_WEIGHTS).to(device)
    loss_fn = build_loss_fn(LOSS_TYPE, class_weights, NUM_CLASSES, DICE_WEIGHT)
    print(f"Using loss function: {LOSS_TYPE} (dice_weight={DICE_WEIGHT})")

    optim = opt.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=2)

    num_workers = min(4, os.cpu_count() or 1)

    train_ds = get_loader(dataset_type="train", batch_size=BATCH_SIZE, num_workers=num_workers)
    val_ds = get_loader(dataset_type="validation", batch_size=BATCH_SIZE, num_workers=num_workers)

    all_tr_losses, all_val_losses = [], []
    all_tr_accs, all_val_accs = [], []
    best_val_loss = float("inf")
    start_epoch = 0
    epochs_without_improvement = 0

    if LAST_CKPT_PATH.exists():
        print(f"Loading existing checkpoint from {LAST_CKPT_PATH} ...")
        checkpoint = torch.load(LAST_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        all_tr_losses = checkpoint["train_losses"]
        all_val_losses = checkpoint["val_losses"]
        all_tr_accs = checkpoint["train_accs"]
        all_val_accs = checkpoint["val_accs"]
        best_val_loss = min(all_val_losses)
        print(f"Resuming training at epoch {start_epoch}.")

    for epoch in range(start_epoch, MAX_EPOCHS):
        tr_loss, tr_acc = train(model, loss_fn, optim, epoch, train_ds, device)
        val_loss, val_acc = validation(model, loss_fn, epoch, val_ds, device)

        scheduler.step(val_loss)

        all_tr_losses.append(tr_loss)
        all_val_losses.append(val_loss)
        all_tr_accs.append(tr_acc)
        all_val_accs.append(val_acc)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "train_losses": all_tr_losses,
            "val_losses": all_val_losses,
            "train_accs": all_tr_accs,
            "val_accs": all_val_accs,
        }, LAST_CKPT_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Epoch {epoch}: new best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"Epoch {epoch}: no improvement ({epochs_without_improvement}/{EARLY_STOP_PATIENCE})")

        print(f"Epoch {epoch} done: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}, lr={optim.param_groups[0]['lr']:.6f}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping after epoch {epoch} — no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs.")
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(all_tr_losses, color="blue", label="Train")
    ax1.plot(all_val_losses, color="red", label="Validation")
    ax1.set_title(f"Loss (combined, dice_weight={DICE_WEIGHT})")
    ax1.legend()

    ax2.plot(all_tr_accs, color="blue", label="Train")
    ax2.plot(all_val_accs, color="red", label="Validation")
    ax2.set_title("Pixel Accuracy")
    ax2.legend()

    plt.savefig(CURVES_PATH)
    plt.show()