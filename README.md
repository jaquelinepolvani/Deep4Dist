# Forest Disturbance Segmentation Using U-Net

A deep learning approach for semantic segmentation of forest disturbances in high-resolution aerial imagery, using the Deep4Dist dataset and a U-Net architecture.

This project was developed for the course *AI Approaches in Earth Observation* (04-GEO-OMA25), taught by Prof. Konstantin Müller, as part of the M.Sc. Applied Earth Observation and Geoanalysis (EAGLE) program at the University of Würzburg.

## Project Overview

Forest disturbances caused by insect outbreaks, drought, and windthrow can lead to significant changes in forest structure. Detecting and mapping these disturbances automatically can support faster and more consistent forest monitoring.

This project investigates the use of a U-Net convolutional neural network for semantic segmentation of different forest disturbance types in high-resolution aerial imagery.

**Objective:** map forest disturbance events in Germany using a deep learning model, and identify the different types of disturbance affecting a given area.

## Dataset

We use the existing **Deep4Dist** dataset, introduced by Rodríguez-Paulino et al. (2026), as our data source.

| Property | Description |
|---|---|
| Location | Rhineland-Palatinate, Germany |
| Number of patches | ~17,500 |
| Patch size | 500 × 500 px (≈ 100 × 100 m at 20 cm resolution) |
| Spatial resolution | 20 cm |
| Input channels | 5 — RGB, near-infrared (NIR), normalized object height (nDSM) |
| Classes | 0 Background · 1 Bark beetle (BB) · 2 Clear-cut (CC) · 3 Windthrow (WT) |
| Training set | 12,147 patches |
| Validation set | 4,470 patches |
| Test set | 905 patches |

The dataset is strongly imbalanced: windthrow represents approximately 1% of all labeled pixels, making this class particularly challenging to detect.

The dataset itself is not included in this repository and can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.14884819).

## Method

### U-Net Architecture

We use a standard U-Net encoder–decoder architecture, adapted to this task with:

- 5 input channels, 4 output classes
- Input size: 512 × 512 pixels
- Four encoder stages, a bottleneck, four decoder stages, and skip connections between encoder and decoder

The U-Net was **trained from scratch**, without pretrained weights. The implementation is based on the publicly available [U-Net implementation by Mateusz Buda](https://github.com/mateuszbuda/brain-segmentation-pytorch), adapted to accept 5-channel input and predict 4 classes.

For context, the original Deep4Dist study benchmarks a **ResU-Net-34**, a U-Net with a ResNet-34 encoder, initialized via transfer learning from weights pretrained on the FLAIR land-cover dataset. Our model uses a simpler, non-pretrained encoder, consistent with the methods covered in our course.

### Data Preparation

- **Normalization:** the 5 input channels are converted to float and scaled to `[0, 1]`.
- **Resizing:** patches are resized from 500×500 to 512×512 px: bilinear interpolation for images, nearest-neighbor for masks (to preserve discrete class labels).
- **Augmentation** (training only): random 90° rotations and horizontal/vertical flips, with the same transform applied to an image and its mask together.

### Handling Class Imbalance

Median-frequency class weighting was used in the main experiments to increase the contribution of underrepresented classes to the loss:

| Class | Weight |
|---|---|
| Background | 0.229 |
| Bark beetle | 1.345 |
| Clear-cut | 0.796 |
| Windthrow | 16.479 |

The first training run (Run 1) instead used simpler, manually chosen weights (1.0 / 2.0 / 2.0 / 4.0), before switching to median-frequency balancing from Run 2 onward.

The loss function combines **weighted Cross-Entropy** with **Dice Loss**. Cross-Entropy evaluates each pixel independently, while Dice Loss evaluates the overlap of the predicted region as a whole. Combining both helps with classes that are both rare and irregularly shaped, such as windthrow.

## Training

The settings below were held constant across all four experimental runs, only the loss composition and augmentation varied between runs (see Experimental Setup).

| Parameter | Setting |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 4 |
| Maximum epochs | 50 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Random seed | 42 |

Early stopping was used to end training once validation performance stopped improving. The best model was selected based on the lowest validation loss.

## Experimental Setup

Four training configurations were evaluated:

| | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| Loss function | CrossEntropyLoss | CombinedLoss (CE + Dice) | CombinedLoss (CE + Dice) | CombinedLoss (CE + Dice) |
| Class weights | BG 1.0 · BB 2.0 · CC 2.0 · WT 4.0 | BG 0.229 · BB 1.345 · CC 0.796 · WT 16.479 | same as Run 2 | same as Run 2 |
| Dice weight | – | 0.5 | 0.4 | 0.5 |
| Augmentation | No | No | Yes (rotation + flips) | Yes (rotation + flips) |
| Early stop patience | 5 | 5 | 5 | 7 |
| Stopped after epoch | 12 | 29 | 26 | 42 |

## Evaluation

Models were evaluated on the independent test set using pixel accuracy, Intersection over Union (IoU), and F1 score, both macro-averaged across the four classes. IoU and F1 give a more informative picture than accuracy alone under strong class imbalance.


## Results

### Overall Performance

Run 4 (weighted CE + Dice, weight 0.5, with augmentation) achieved the best overall performance:

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Paper* |
|---|---|---|---|---|---|
| Accuracy | 79.5% | 85.8% | 85.4% | **85.9%** | 88.2% |
| Mean IoU | 45.0% | 63.9% | 62.5% | **65.0%** | 70.3% |
| Mean F1 | 55.6% | 76.6% | 75.0% | **77.6%** | 81.9% |

*Reported by Rodríguez-Paulino et al. (2026) for their pretrained ResU-Net-34; our U-Net was trained from scratch and is not directly equivalent.

### Per-Class IoU

| Class | Run 4 IoU |
|---|---|
| Background | 81.0% |
| Bark beetle | 74.4% |
| Clear-cut | 64.3% |
| Windthrow | 40.2% |

Windthrow IoU increased from 0.00 (Run 1) to 0.40 (Run 4), though it remains the hardest class to segment. Background remains the strongest and most stable class across all four runs.

### Visual Comparison

![Model comparison across runs](figures/model_comparison_3mixed_samples.png)

RGB input, ground truth, and predictions from all four runs on test patches containing a mix of disturbance classes, illustrating how the different training strategies affect the spatial delineation of forest disturbances.

## Discussion

### Main Findings

- Run 4 achieved the best overall performance (85.9% accuracy, 65.0% macro IoU)
- Windthrow improved: IoU from 0.00 (Run 1) to 0.40 (Run 4)

### Possible Future Improvements

- Try a ResNet-34 encoder with pretraining, or alternative architectures
- Use padding instead of resizing (500→512) to avoid interpolation artifacts
- Stronger/more augmentation (e.g. channel dropout, as described in the paper)
- Try other loss functions or loss function combinations
- Separate binary model just for windthrow, combined with the main model
- Larger batch size (currently limited by 6 GB VRAM)
- Ensemble of multiple trained models

### Limitations

- Severe class imbalance remains a major challenge, especially for the rare windthrow class.
- Trained entirely from scratch: no benefit from pretrained transfer learning.
- No independent geographic test region: all patches come from Rhineland-Palatinate.

## Authors

- Jaqueline Lopes Polvani — jaqueline.lopes-polvani@stud-mail.uni-wuerzburg.de
- Jolien Grafe — jolien.grafe@stud-mail.uni-wuerzburg.de

M.Sc. Applied Earth Observation and Geoanalysis (EAGLE), University of Würzburg

## References

Rodríguez-Paulino, E., Stoffels, J., Schlerf, M., Röder, A., Wagner, A., & Udelhoven, T. (2026). An AI-ready remote sensing dataset for high-resolution forest disturbance mapping. *Scientific Data*, 13, 490. https://doi.org/10.1038/s41597-026-07084-8

Buda, M. *brain-segmentation-pytorch* [Source code]. GitHub. https://github.com/mateuszbuda/brain-segmentation-pytorch
