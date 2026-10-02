# Web-Supervised Monocular Relative Depth Estimation

A PyTorch implementation of **monocular relative depth estimation** trained on the **ReDWeb V1** dataset. The project predicts a dense relative-depth map from a single RGB image using a weakly supervised **ResNeXt-101 32×8d** encoder, multi-scale residual feature fusion, and a pairwise ranking objective.

Unlike metric-depth systems, this project does **not** attempt to recover depth in meters. Its goal is to learn the **ordinal structure of a scene**: which pixels or regions are closer, farther away, or at approximately the same relative depth.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Data Preparation and Augmentation](#data-preparation-and-augmentation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [RGB and Relative-Depth Samples](#1-rgb-and-relative-depth-samples)
  - [RGB Resolution Distribution](#2-rgb-resolution-distribution)
  - [Global Depth-Value Distribution](#3-global-depth-value-distribution)
  - [Per-Image Depth Histograms](#4-per-image-depth-histograms)
  - [RGB–Depth Overlays](#5-rgbdepth-overlays)
  - [Vertical Depth Profile](#6-vertical-depth-profile)
  - [RGB Edges vs. Depth Discontinuities](#7-rgb-edges-vs-depth-discontinuities)
- [Model Architecture](#model-architecture)
- [Pairwise Ranking Loss](#pairwise-ranking-loss)
- [Training Configuration](#training-configuration)
- [Training Results](#training-results)
- [Qualitative Inference](#qualitative-inference)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Checkpointing](#checkpointing)
- [Implementation Notes and Limitations](#implementation-notes-and-limitations)
- [Possible Improvements](#possible-improvements)
- [References](#references)
- [License and Dataset Usage](#license-and-dataset-usage)

---

## Project Overview

Monocular depth estimation is inherently ambiguous because a single RGB image does not provide direct geometric triangulation. Instead of predicting absolute metric distance, this project learns a **relative depth representation** from dense relative-depth supervision.

Given an RGB image \(I\), the model outputs a single-channel map

\[
\hat{D} = f_\theta(I),
\]

where larger or smaller predicted values encode relative ordering between scene points. Training is driven by randomly sampled pixel pairs rather than by direct per-pixel regression alone.

The notebook implements the complete workflow:

1. Load paired RGB images and relative-depth maps from ReDWeb V1.
2. Split the 3,600 samples into training, validation, and test subsets.
3. Apply paired geometric augmentation to RGB and depth.
4. Explore RGB resolutions, depth-value statistics, spatial alignment, depth profiles, and boundary structure.
5. Build a ResNeXt-based encoder-decoder depth network.
6. Optimize the network using a pairwise ordinal ranking loss with hard-pair mining.
7. Save and reload model checkpoints.
8. Visualize predicted depth maps against RGB inputs and ground-truth relative depth.

---

## Dataset

This project uses **ReDWeb V1 (Relative Depth from Web)**, introduced in *Monocular Relative Depth Perception with Web Stereo Data Supervision*.

ReDWeb V1 contains **3,600 RGB / relative-depth pairs** spanning diverse indoor and outdoor scenes. The dataset provides dense relative-depth supervision generated from web stereo imagery.

### Expected directory structure

```text
ReDWeb_V1/
├── Imgs/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── RDs/
    ├── image_1.png
    ├── image_2.png
    └── ...
```

The notebook constructs RGB/depth pairs by matching the RGB filename stem to a `.png` file in `RDs/`.

### Dataset split

The notebook uses a deterministic random split with `random_state=42`:

| Split | Percentage | Number of images |
|---|---:|---:|
| Training | 70% | 2,520 |
| Validation | 15% | 540 |
| Test | 15% | 540 |
| **Total** | **100%** | **3,600** |

The first split holds out 30% of the dataset, and that subset is then divided evenly into validation and test sets.

---

## Pipeline

```text
RGB image
   │
   ├── paired augmentation
   │   ├── optional brightness variation
   │   ├── optional horizontal flip
   │   └── random resized crop
   │
   ▼
384 × 384 RGB tensor
   │
   ▼
WSL ResNeXt-101 32×8d encoder
   │
   ├── stage 1: 256 channels
   ├── stage 2: 512 channels
   ├── stage 3: 1024 channels
   └── stage 4: 2048 channels
   │
   ▼
3×3 skip projections → 256 channels at every scale
   │
   ▼
Residual multi-scale feature-fusion decoder
   │
   ▼
Adaptive output head
   │
   ▼
384 × 384 relative-depth prediction
   │
   ▼
Random pixel-pair sampling
   │
   ▼
Ordinal ranking loss + hard-pair mining
```

---

## Data Preparation and Augmentation

The custom `ReDWebDataset` loads each RGB image with OpenCV and its matching relative-depth image with `cv2.IMREAD_UNCHANGED`.

### Training transform

For each training sample, the notebook applies:

- **Brightness augmentation:** 50% probability.
  - Converts the image through HSV.
  - Multiplies the value channel by a random factor in `[0.9, 1.1]`.
- **Horizontal flip:** 50% probability.
  - The exact same flip is applied to RGB and depth so correspondence is preserved.
- **Random resized crop:**
  - A random scale in `[0.8, 1.0]` is selected.
  - The same crop is applied to both RGB and depth.
- **Resize:** both modalities are resized to `384 × 384`.
- **Normalization:** RGB and depth values are divided by `255`.

This paired augmentation is important because any geometric transform applied only to the RGB image would destroy pixel-wise RGB/depth alignment.

### Validation and test transform

Validation and test data are processed deterministically:

```text
resize to 384 × 384
        ↓
convert to tensor
        ↓
divide values by 255
```

No random crop or horizontal flip is used for validation/test samples.

---

# Exploratory Data Analysis

The notebook includes several complementary analyses. Together they inspect the dataset at three levels:

- **global statistics** — image resolution and depth-value distributions;
- **sample-level structure** — per-image histograms and RGB/depth overlays;
- **spatial structure** — vertical depth profiles and boundary alignment.

## 1. RGB and Relative-Depth Samples

![RGB and relative-depth samples](visualize_samples.png)

This visualization shows randomly selected RGB images alongside their corresponding relative-depth maps after preprocessing.

### What this plot checks

- RGB and depth maps are paired correctly.
- Resizing preserves the overall scene structure.
- Relative-depth maps are dense rather than sparse point annotations.
- The dataset contains substantial scene diversity, including people, architecture, objects, streets, and other unconstrained imagery.

The relative-depth maps are smooth over many surfaces while still preserving large scene-level depth transitions. Because these maps encode **relative** rather than calibrated metric depth, the grayscale intensity should be interpreted as scene ordering rather than physical distance in meters.

---

## 2. RGB Resolution Distribution

![RGB width and height distributions](RGB_Distribution.png)

The notebook inspects up to the first 500 samples and records their native width and height before the fixed `384 × 384` model resize.

The most common observed RGB resolutions were:

| Resolution | Count in inspected subset |
|---|---:|
| `484 × 356` | 24 |
| `772 × 572` | 10 |
| `772 × 581` | 6 |
| `484 × 366` | 6 |
| `484 × 371` | 5 |

The corresponding depth maps have the same most-common native resolutions, supporting the expected spatial correspondence between RGB and relative-depth data.

### Interpretation

The width distribution is strongly multi-modal rather than concentrated around a single resolution. Heights are also variable. This confirms that the original dataset is **heterogeneous in image size and aspect ratio**.

A fixed network input size therefore requires explicit resizing or crop/resize preprocessing. In this project, every sample ultimately becomes `384 × 384`.

### Practical consequence

Direct resizing to a square can distort the original aspect ratio. The current implementation favors simplicity and fixed tensor sizes; an alternative implementation could preserve aspect ratio and use padding or scale-aware cropping.

---

## 3. Global Depth-Value Distribution

![Depth value distribution](Depth_Value_Distribution.png)

The notebook samples up to 200 depth maps, removes zero-valued pixels for this particular analysis, concatenates the remaining pixels, and plots the global distribution.

Recorded non-zero statistics:

| Statistic | Value |
|---|---:|
| Minimum | `1.0000` |
| Maximum | `255.0000` |
| Mean | `160.5413` |
| Standard deviation | `72.1454` |

### Interpretation

The distribution spans nearly the full 8-bit range and is not uniform. The pronounced accumulation around `255` indicates that many pixels are encoded at the largest available relative-depth value.

This plot is useful for identifying:

- saturation at the upper encoding boundary;
- whether depth values occupy a narrow or broad range;
- whether zeros should be handled separately;
- whether a simple `/255` normalization is numerically appropriate.

For training, the notebook divides depth maps by `255`, mapping the encoded depth range approximately into `[0,1]`.

> **Important:** these values represent the dataset's relative-depth encoding. They should not be interpreted as centimeters, meters, or any other metric unit.

---

## 4. Per-Image Depth Histograms

![Per-image relative-depth histograms](Per-image_depth_histograms.png)

A global histogram can hide the structure of individual scenes, so the notebook also plots several RGB images next to their own non-zero depth distributions.

### What the examples show

Different scenes have very different relative-depth signatures:

- architectural scenes may form distinct depth modes corresponding to facade, foreground, and background;
- cluttered mechanical scenes can create several separate peaks;
- scenes with multiple people or objects at different positions can show highly multi-modal distributions;
- some images contain a large mass at the maximum encoded value.

### Why this matters

A single global normalization scheme is applied across scenes whose depth distributions are highly heterogeneous. This is one reason a **pairwise ordinal objective** is attractive: learning the ordering of points is less dependent on absolute scale than conventional pixel-wise metric regression.

---

## 5. RGB–Depth Overlays

![RGB/depth overlay examples](overlay_depth_grid.png)

For these visualizations, each depth map is resized to the RGB image resolution when necessary, normalized by its per-image maximum, rendered with the `inferno` colormap, and alpha-blended over the RGB image.

### Purpose

This is primarily a **registration sanity check**.

A correct RGB/depth pair should show coherent depth regions that follow meaningful scene structure. For example:

- an isolated foreground object should have a depth region that follows its shape;
- a person or animal should be separated from the surrounding surface when depth differs;
- railings, large objects, and landscape regions should exhibit spatially plausible transitions.

### What the examples suggest

The overlays show that major object and scene regions generally align with depth structure. They also demonstrate that relative depth is not simply an RGB saliency map: visually prominent color or texture changes do not automatically imply a depth change.

---

## 6. Vertical Depth Profile

![Vertical relative-depth profile](Vertical_depth_profile_example.png)

To summarize how depth changes from the top of an image to the bottom, the notebook divides a depth map into horizontal strips and computes the mean non-zero depth value in each strip.

For an image with height \(H\), each strip \(S_k\) contributes approximately

\[
\bar{d}_k =
\frac{1}{|S_k^{+}|}
\sum_{p \in S_k^{+}} d_p,
\]

where \(S_k^{+}\) contains only non-zero depth pixels.

The resulting mean values are plotted against vertical pixel position.

### Why this visualization is useful

Many natural images exhibit large-scale vertical depth organization. For example:

- sky or distant background often appears toward the top;
- intermediate scene elements occupy the middle;
- nearby ground or foreground objects often occupy lower regions.

The shown example contains a strong transition in the middle/lower portion of the image, demonstrating that the depth map captures large spatial changes rather than behaving like uniform noise.

This is an exploratory diagnostic, not a training target: the model is not explicitly forced to follow a vertical-depth prior.

---

## 7. RGB Edges vs. Depth Discontinuities

![RGB edges compared with depth gradients](Depth_edge_vs_RGB_edge_example.png)

This analysis compares three views of the same scene:

1. the original RGB image;
2. RGB edges detected with **Canny**;
3. the magnitude of the depth-map gradient computed using **Sobel derivatives**.

For the depth map \(D\), the notebook computes

\[
G_x = \frac{\partial D}{\partial x},
\qquad
G_y = \frac{\partial D}{\partial y},
\]

and visualizes

\[
|\nabla D| = \sqrt{G_x^2 + G_y^2}.
\]

### Interpretation

Strong depth gradients frequently coincide with important object boundaries or geometry transitions. In the bird example, the main silhouette is visible in both the RGB-edge map and depth-gradient representation.

However, RGB and depth boundaries should **not** match perfectly:

- RGB contains texture, markings, shadows, and illumination edges that may lie on a single surface;
- a depth discontinuity can occur where RGB contrast is weak;
- fine appearance details may generate Canny edges without any geometric change.

This comparison therefore provides a useful qualitative check that depth boundaries are structurally meaningful rather than simply reproducing all image edges.

---

# Model Architecture

The network follows an encoder-decoder design with multi-scale skip features.

## 1. Encoder: WSL ResNeXt-101 32×8d

The encoder is loaded through:

```python
torch.hub.load(
    "facebookresearch/WSL-Images",
    "resnext101_32x8d_wsl"
)
```

The classifier head is not used. Instead, four progressively lower-resolution feature stages are extracted.

| Encoder stage | Output channels |
|---|---:|
| Layer 1 | 256 |
| Layer 2 | 512 |
| Layer 3 | 1024 |
| Layer 4 | 2048 |

The first encoder stage also contains the original convolution, batch normalization, ReLU, max-pooling, and ResNeXt layer 1.

---

## 2. Skip projections

Each encoder output is converted to **256 channels** with a `3 × 3` convolution:

```text
256  → 256
512  → 256
1024 → 256
2048 → 256
```

Using a common channel dimension makes it possible to fuse features from different scales in a uniform decoder.

---

## 3. Residual convolution block

The decoder uses a custom residual block:

```text
input
  │
 ReLU
  │
3×3 Conv
  │
 ReLU
  │
3×3 Conv
  │
  +──────── original input
  │
output
```

For input \(x\), the block computes

\[
y = x + F(x),
\]

where \(F\) contains two `3 × 3` convolutions with ReLU activations.

Residual refinement helps preserve incoming information while allowing the decoder to learn corrections to the feature representation.

---

## 4. Feature-fusion block

Each `FeatureFusionBlock` combines information from adjacent encoder/decoder scales.

When two inputs are supplied:

1. the skip feature is refined by a residual block;
2. it is added to the main decoder feature;
3. the fused tensor passes through another residual block;
4. the result is bilinearly upsampled by a factor of two.

Conceptually:

```text
coarse decoder feature ───────────┐
                                  + → residual refinement → ×2 upsample
encoder skip → residual block ────┘
```

Four feature-fusion blocks progressively reconstruct spatial detail from the deepest encoder representation toward the input resolution.

---

## 5. Output head

After the final fusion stage, the model applies:

```text
256 channels
    ↓ 3×3 Conv
128 channels
    ↓ ×2 bilinear interpolation
128 channels
    ↓ 3×3 Conv + ReLU
32 channels
    ↓ 1×1 Conv
1-channel relative-depth map
```

For a `384 × 384` input, the final output is a single-channel `384 × 384` prediction.

### Model size

The notebook's `torchsummary` output reports:

| Property | Value |
|---|---:|
| Total parameters | `104,182,785` |
| Trainable parameters | `104,182,785` |
| Non-trainable parameters | `0` |
| Parameter memory estimate | ~`397.43 MB` |

All encoder and decoder parameters are trainable in the current implementation.

---

# Pairwise Ranking Loss

Relative depth is learned through randomly sampled pairs of pixels.

For every image in a batch, the notebook samples `1,000` random pixel pairs by default:

\[
(i,j).
\]

Let the ground-truth normalized relative-depth values be \(g_i\) and \(g_j\), and model predictions be \(p_i\) and \(p_j\).

## Ordinal relation assignment

A tolerance

\[
\sigma = 0.02
\]

is used to determine whether two points should be treated as meaningfully different.

The pair label is:

\[
l_{ij} =
\begin{cases}
1, & \frac{g_i}{g_j} > 1+\sigma \\
-1, & \frac{g_j}{g_i} > 1+\sigma \\
0, & \text{otherwise}
\end{cases}
\]

Thus each pair represents one of three relations:

- point \(i\) has greater relative depth;
- point \(j\) has greater relative depth;
- the two points are approximately equal.

## Equal-depth pairs

When \(l_{ij}=0\), the loss penalizes disagreement directly:

\[
L_{\text{equal}} = (p_i - p_j)^2.
\]

## Unequal-depth pairs

For unequal pairs, the notebook uses a logistic ranking term:

\[
L_{\text{rank}}
=
\log\left(
1 + \exp\left((p_j-p_i)l_{ij}\right)
\right).
\]

This penalizes predictions whose ordering conflicts with the ground-truth ordinal relation.

## Hard-pair mining

After computing losses for unequal pairs, they are sorted and the lowest-loss **25%** are discarded.

Only the harder **75%** of unequal pairs contribute to the final loss.

```text
random unequal pairs
       ↓
compute ranking loss
       ↓
sort by loss
       ↓
discard easiest 25%
       ↓
optimize on harder 75%
```

This focuses learning on pairs whose ordering is not already handled confidently by the model.

---

# Training Configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Input size | `384 × 384` |
| Training batch size | `4` |
| Validation batch size | `4` |
| Test batch size | `8` |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss | Pairwise ranking loss |
| Random pairs/image | `1,000` by default |
| Ordinal tolerance `σ` | `0.02` |
| Hard-pair strategy | Drop easiest 25% of unequal pairs |
| Epochs | `7` |
| Hardware path | CUDA / GPU |
| Training shuffle | Yes |
| Validation shuffle | No |

The current notebook moves the model and batches directly to CUDA with `.cuda()`, so a GPU is expected unless the code is adapted to use a general `device`.

---

# Training Results

The notebook records the following training and validation losses:

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 1682.0791 | 1515.0409 |
| 2 | 1590.5203 | 1511.4130 |
| 3 | 1547.6383 | 1507.8587 |
| 4 | 1513.9415 | **1431.0780** |
| 5 | 1467.9698 | 1442.8243 |
| 6 | 1433.7750 | 1434.2833 |
| 7 | **1412.5344** | 1434.8410 |

### Observations

- Training loss decreases consistently across all seven epochs.
- Validation loss improves substantially through epoch 4.
- The lowest recorded validation loss is **1431.0780 at epoch 4**.
- Validation loss then fluctuates slightly while training loss continues to fall.
- The current notebook saves the model after the final epoch rather than automatically restoring the best-validation epoch.

These loss magnitudes are specific to the implemented pair-sampling and loss-aggregation procedure; they should not be interpreted as a standard metric such as RMSE or WHDR.

The notebook does **not** currently report a formal test-set benchmark such as WHDR, AbsRel, RMSE, or \(\delta\)-accuracy. Accordingly, no benchmark claim is made here.

---

# Qualitative Inference

The final inference code displays, for samples from the test loader:

```text
RGB input | Ground-truth relative depth | Predicted relative depth
```

This is the appropriate first qualitative check for a dense prediction model because it reveals whether the network recovers:

- global foreground/background ordering;
- object-level depth separation;
- broad surface continuity;
- meaningful scene boundaries.

For rigorous model assessment, this qualitative comparison should be complemented by a relative-depth metric such as WHDR and, where appropriate, boundary or ranking accuracy.

---

# Repository Structure

A minimal repository layout for the current project is:

```text
.
├── cv-project.ipynb
├── README.md
├── visualize_samples.png
├── RGB_Distribution.png
├── Depth_Value_Distribution.png
├── Per-image_depth_histograms.png
├── overlay_depth_grid.png
├── Vertical_depth_profile_example.png
└── Depth_edge_vs_RGB_edge_example.png
```

The ReDWeb dataset itself should remain outside version control unless its distribution terms explicitly permit otherwise.

---

# Installation

A Python environment with CUDA-enabled PyTorch is recommended.

```bash
pip install torch torchvision
pip install numpy matplotlib opencv-python
pip install scikit-learn tqdm torchsummary
```

The notebook also downloads the pretrained WSL ResNeXt model through `torch.hub`, so network access is required the first time the encoder weights are loaded.

### Main dependencies

- Python 3
- PyTorch
- TorchVision
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- tqdm
- torchsummary

---

# Running the Notebook

## 1. Prepare ReDWeb V1

Place the dataset in a directory containing:

```text
Imgs/
RDs/
```

The notebook currently uses:

```python
data_path = Path("/kaggle/input/dataset-redweb/ReDWeb_V1")
```

Change this path if your local or cloud environment stores the dataset elsewhere.

## 2. Run dataset creation and splitting

The notebook creates a `ReDWebDataset`, builds matching RGB/depth filename lists, and produces the 70/15/15 split.

## 3. Run EDA

Execute the analysis cells to reproduce:

- image-size histograms;
- global relative-depth statistics;
- per-image depth histograms;
- RGB/depth overlays;
- vertical depth profiles;
- RGB-edge/depth-gradient comparisons.

## 4. Build the model

The encoder weights are loaded from Facebook Research's WSL model repository through `torch.hub`.

## 5. Train

The notebook trains for seven epochs with:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = ranking_loss
```

## 6. Save and evaluate

After training, save the checkpoint and visualize test predictions against their ground-truth relative-depth maps.

---

# Checkpointing

The notebook currently saves:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": criterion,
}, "checkpoint.pth")
```

## PyTorch 2.6+ note

The recorded notebook output shows that reloading this checkpoint can raise an `UnpicklingError` in PyTorch 2.6+ because the checkpoint contains the Python function object `ranking_loss`, while `torch.load()` now defaults to safer weight-only loading behavior.

A more portable checkpoint is:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss_name": "ranking_loss",
}, "checkpoint.pth")
```

and:

```python
checkpoint = torch.load("checkpoint.pth", weights_only=True)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

epoch = checkpoint["epoch"]
```

Avoid serializing executable Python functions when only their name/configuration is required.

---

# Implementation Notes and Limitations

The notebook is a working experimental implementation, but several details are important for reproducibility and future refinement.

### 1. RGB channel order

`cv2.imread()` loads images in **BGR** order.

The current training transformation converts BGR → RGB only inside the optional brightness-augmentation branch, while the validation transform does not explicitly convert BGR → RGB.

For consistent pretrained-encoder input, convert every image explicitly:

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

before augmentation/tensor conversion.

### 2. Pretrained encoder normalization

The WSL ResNeXt weights were released for RGB images using ImageNet-style normalization. The current notebook scales RGB values to `[0,1]` but does not subtract the pretrained mean or divide by the pretrained standard deviation.

A standard normalization step would use:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

after conversion to RGB and `[0,1]`.

Changing this preprocessing can alter training behavior, so results should be re-evaluated after the correction.

### 3. Invalid/zero depth pixels

The EDA explicitly removes zero-valued depth pixels, but the current ranking-loss sampler does not mask zeros before drawing pixel pairs.

Because the ordinal relation uses ratios such as `g_i / g_j`, zero values can create undefined or infinite ratios.

A stronger implementation should construct a valid-depth mask and sample pairs only from valid pixels.

### 4. Square resizing

All samples are resized to `384 × 384`, regardless of original aspect ratio. This simplifies batching but may introduce geometric distortion.

Potential alternatives include:

- resize while preserving aspect ratio and pad;
- random crop after aspect-ratio-preserving scaling;
- train with multiple resolutions.

### 5. Depth interpolation

Training augmentation resizes depth maps with OpenCV's default interpolation, while some EDA functions use nearest-neighbor interpolation.

For ordinal/discrete relative-depth encodings, interpolation strategy can affect label values near boundaries and should be chosen deliberately.

### 6. Loss scaling

The ranking loss sums a large number of pair losses and then the training loop averages only across batches. It is therefore expected to have a large numerical magnitude.

Normalizing by the number of valid pairs would produce a loss whose scale is easier to compare across different pair counts and batch sizes.

### 7. Best-model selection

The best validation loss occurs at epoch 4, but the notebook saves the final epoch.

A production training loop should save a checkpoint whenever validation performance improves.

### 8. No formal test metric yet

The current implementation focuses on training/validation ranking loss and qualitative visualization. A full experimental report should include an accepted relative-depth evaluation metric on the held-out test set.

---

# Possible Improvements

Several extensions would make the project more robust and research-ready:

1. **Fix preprocessing consistency**
   - explicit BGR → RGB conversion;
   - apply the normalization expected by the pretrained WSL encoder.

2. **Mask invalid depth values**
   - exclude zero/invalid pixels from pair sampling.

3. **Vectorize the ranking loss**
   - remove the Python loop over individual pixel pairs;
   - improve GPU utilization and training speed.

4. **Normalize the pairwise loss**
   - divide by the number of contributing pairs.

5. **Use validation-based checkpointing**
   - save the best model instead of only the final epoch.

6. **Add a learning-rate scheduler**
   - e.g. cosine decay or `ReduceLROnPlateau`.

7. **Add quantitative evaluation**
   - WHDR / ordinal accuracy;
   - pairwise ranking accuracy;
   - optional edge-aware depth metrics.

8. **Preserve aspect ratio**
   - padding or multi-scale training instead of unconditional square warping.

9. **Add mixed-precision training**
   - `torch.cuda.amp` / `torch.autocast` can reduce memory use and speed up training on supported GPUs.

10. **Make device handling portable**
    - support CUDA, Apple Silicon, or CPU through `torch.device`.

11. **Add reproducibility controls**
    - seed Python, NumPy, and PyTorch RNGs;
    - record package versions and GPU information.

12. **Add experiment tracking**
    - save train/validation curves, hyperparameters, and qualitative predictions for each run.

---

# References

### ReDWeb

Ke Xian, Chunhua Shen, Zhiguo Cao, Hao Lu, Yang Xiao, Ruibo Li, and Zhenbo Luo.  
**Monocular Relative Depth Perception with Web Stereo Data Supervision.**  
CVPR, 2018.

- Project page: https://sites.google.com/site/redwebcvpr18/
- Paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Xian_Monocular_Relative_Depth_CVPR_2018_paper.html

```bibtex
@inproceedings{Xian_2018_CVPR,
  title     = {Monocular Relative Depth Perception with Web Stereo Data Supervision},
  author    = {Xian, Ke and Shen, Chunhua and Cao, Zhiguo and Lu, Hao and
               Xiao, Yang and Li, Ruibo and Luo, Zhenbo},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}
```

### Weakly Supervised ResNeXt Pretraining

Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten.  
**Exploring the Limits of Weakly Supervised Pretraining.**  
ECCV, 2018.

- WSL-Images repository: https://github.com/facebookresearch/WSL-Images
- Paper: https://arxiv.org/abs/1805.00932

```bibtex
@inproceedings{mahajan2018exploring,
  title     = {Exploring the Limits of Weakly Supervised Pretraining},
  author    = {Mahajan, Dhruv and Girshick, Ross and Ramanathan, Vignesh and
               He, Kaiming and Paluri, Manohar and Li, Yixuan and
               Bharambe, Ashwin and van der Maaten, Laurens},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2018}
}
```

---

# License and Dataset Usage

This repository's source code does not currently include an explicit software license.

The external resources used by the project have their own terms:

- **ReDWeb V1** is provided for **research use** according to the dataset project page.
- The Facebook Research **WSL-Images** pretrained models are released under their repository's stated license and usage conditions.

Before redistributing the dataset, pretrained weights, or derivatives, review the corresponding upstream licenses and terms.

---

## Summary

This project demonstrates an end-to-end workflow for dense **monocular relative-depth estimation in unconstrained scenes**:

- a diverse 3,600-pair ReDWeb dataset;
- paired RGB/depth augmentation;
- extensive dataset diagnostics;
- a WSL-pretrained ResNeXt-101 encoder;
- a residual, multi-scale feature-fusion decoder;
- random ordinal-pair supervision;
- hard-pair ranking loss;
- GPU training and checkpointing;
- qualitative prediction analysis.

The current implementation provides a solid experimental baseline while also exposing several clear directions for improving preprocessing consistency, loss robustness, quantitative evaluation, and reproducibility.
