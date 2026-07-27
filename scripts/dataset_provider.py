"""
Dataset provider for the Deep4Dist forest-disturbance dataset.

Expected folder layout (rooted at BASE_PATH, defined in config.py):

    BASE_PATH/
        train/
            image/*.tif
            mask/*.tif
        validation/
            image/*.tif
            mask/*.tif
        test/
            image/*.tif
            mask/*.tifpixi

Each image is a 5-band GeoTIFF (RGB + NIR + normalized DSM), uint8.
Each mask is a 1-band GeoTIFF with integer class labels (0-3).
"""

import os
from pathlib import Path

import torch
import torch.utils.data as td
import rasterio
import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from config import BASE_PATH

# Resize is used to make the 500x500 input compatible with this U-Net:
# the encoder has 4 pooling steps (halving each time), so height/width
# must be divisible by 2^4 = 16. 512 is the nearest such "clean" size.
#
# Image and mask need DIFFERENT interpolation modes:
# - Image: bilinear (default) is fine, smooths pixel values slightly.
# - Mask: MUST be NEAREST, otherwise interpolation would blend class
#   indices into invalid, non-integer "in-between" classes.
image_transform = transforms.Compose([
    transforms.Resize((512, 512))
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST)
])


class Deep4DistDataset(td.Dataset):
    """
    Lazy-loading Dataset for Deep4Dist.

    Lazy loading means: __init__ only records file paths (fast, tiny
    memory footprint). The actual reading + preprocessing of each
    image/mask pair happens in __getitem__, i.e. only when the
    DataLoader actually requests that sample. This is necessary here
    because loading all ~12k train images eagerly into RAM would need
    60+ GB of memory.
    """

    def __init__(self, dataset_type, limit=None):
        """
        Args:
            dataset_type: one of "train", "validation", "test".
            limit: optional int, only use the first N samples
                   (useful for quick smoke tests).
        """
        image_dir = BASE_PATH / dataset_type / "image"
        mask_dir = BASE_PATH / dataset_type / "mask"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Expected image folder not found: {image_dir}\n"
                f"Check that BASE_PATH in config.py points to the correct "
                f"Deep4Dist folder, and that the '{dataset_type}/image' "
                f"subfolder exists there."
            )

        image_files = sorted(image_dir.glob("*.tif"))
        if limit is not None:
            image_files = image_files[:limit]

        # Only store (image_path, mask_path) pairs here — NOT the actual
        # pixel data. This keeps __init__ fast and memory-light even for
        # the full ~12k-image train set.
        self.samples = []
        skipped = 0

        for image_file in tqdm(image_files, desc=f"Indexing {dataset_type} data"):
            # Image and mask share the exact same filename, just live in
            # different subfolders (image/ vs mask/).
            mask_file = mask_dir / image_file.name

            if not mask_file.exists():
                # Guards against incomplete downloads/extractions: skip
                # any image that doesn't have a matching mask, instead of
                # crashing mid-training.
                skipped += 1
                continue

            self.samples.append((image_file, mask_file))

        if skipped > 0:
            print(
                f"WARNING: {skipped} of {len(image_files)} images in "
                f"'{dataset_type}' had no matching mask and were skipped. "
                f"If this number is large, your download/extraction may "
                f"be incomplete — consider re-extracting the data."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_file, mask_file = self.samples[index]

        # rasterio reads GeoTIFFs directly as (bands, height, width),
        # so no manual transpose is needed here (unlike skimage.io,
        # which returns (height, width, channels)).
        with rasterio.open(image_file) as src:
            img_read = src.read()  # shape: (5, 500, 500), dtype uint8

        with rasterio.open(mask_file) as src:
            mask_read = src.read()  # shape: (1, 500, 500), dtype uint8

        # Image: cast to float and scale to [0, 1]. All 5 bands are
        # already stored as uint8 (0-255), including the height channel,
        # so a single uniform scaling works for every band here.
        img_read = torch.FloatTensor(img_read.astype(np.float32)) / 255.0

        # Mask: cast to Long (int64) WITHOUT normalizing. These are
        # categorical class indices (0=background, 1-3=disturbance
        # types), not intensities — dividing by 255 would destroy them.
        # LongTensor is required by nn.CrossEntropyLoss.
        mask_read = torch.LongTensor(mask_read.astype(np.int64))

        img_read = image_transform(img_read)
        mask_read = mask_transform(mask_read)

        return img_read, mask_read


def get_loader(dataset_type, batch_size, limit=None, num_workers=None):
    """
    Builds a DataLoader for the given split ("train", "validation", "test").

    num_workers: number of background processes used to load/preprocess
    data in parallel while the GPU is busy computing the current batch.
    Defaults to min(4, available CPU cores) so it doesn't over-subscribe
    a machine with fewer cores than the original development machine.
    """
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    dataset = Deep4DistDataset(dataset_type, limit=limit)

    return td.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,                     # faster CPU -> GPU transfer
        persistent_workers=(num_workers > 0),  # keep workers alive across epochs
    )