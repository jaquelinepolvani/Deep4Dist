"""
Computes per-class pixel frequencies across the full train set,
and derives class weights for CrossEntropyLoss using both the
inverse-frequency and median-frequency-balancing methods.
"""

import numpy as np
from tqdm import tqdm

from config import BASE_PATH

NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Bark Beetle", "Clear-cut", "Windthrow"]


if __name__ == '__main__':
    mask_dir = BASE_PATH / "train" / "mask"
    mask_files = sorted(mask_dir.glob("*.tif"))

    import rasterio
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for mask_file in tqdm(mask_files, desc="Counting class pixels"):
        with rasterio.open(mask_file) as src:
            mask = src.read()
        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
        class_pixel_counts += counts[:NUM_CLASSES]

    total_pixels = class_pixel_counts.sum()
    frequencies = class_pixel_counts / total_pixels

    # Method 1: inverse frequency
    inverse_freq_weights = total_pixels / (NUM_CLASSES * class_pixel_counts)

    # Method 2: median frequency balancing
    median_freq = np.median(frequencies)
    median_freq_weights = median_freq / frequencies

    print(f"{'Class':<15} {'Pixel count':>15} {'Frequency':>12} {'Inv-Freq W':>12} {'Med-Freq W':>12}")
    for name, count, freq, w1, w2 in zip(
        CLASS_NAMES, class_pixel_counts, frequencies, inverse_freq_weights, median_freq_weights
    ):
        print(f"{name:<15} {count:>15,} {freq:>12.4%} {w1:>12.3f} {w2:>12.3f}")

    print(f"\nSuggested CLASS_WEIGHTS (inverse frequency):        {list(np.round(inverse_freq_weights, 3))}")
    print(f"Suggested CLASS_WEIGHTS (median frequency balancing): {list(np.round(median_freq_weights, 3))}")