import rasterio
import numpy as np
import glob
import os


img_path = r"D:\EAGLE\Deep_Learning_for_EO\Deep4dist\train\image\dop20rgbi_32_298_5548_2_rp_2023_006_020.tif"
mask_path = r"D:\EAGLE\Deep_Learning_for_EO\Deep4dist\train\mask\dop20rgbi_32_298_5548_2_rp_2023_006_020.tif"

with rasterio.open(img_path) as src:
    img = src.read()
    print("IMAGE shape:", img.shape, "dtype:", img.dtype)
    print("IMAGE min/max je Band:", [(img[b].min(), img[b].max()) for b in range(img.shape[0])])

with rasterio.open(mask_path) as src:
    mask = src.read()
    print("MASK shape:", mask.shape, "dtype:", mask.dtype)
    print("MASK unique values:", np.unique(mask))


mask_dir = r"D:\EAGLE\Deep_Learning_for_EO\Deep4dist\train\mask"
mask_files = glob.glob(os.path.join(mask_dir, "*.tif"))

print(f"Anzahl Maskendateien: {len(mask_files)}")

all_unique_values = set()
for mf in mask_files:
    with rasterio.open(mf) as src:
        mask = src.read()
        all_unique_values.update(np.unique(mask).tolist())

print("Alle vorkommenden Klassenwerte im gesamten validation-Set:", sorted(all_unique_values))