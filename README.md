# Mars Terrain Semantic Segmentation — Deepmindset

Group project for the *Artificial Neural Networks and Deep Learning* course at
Politecnico di Milano (M.Sc., A.Y. 2024/25). The task is **pixel-wise
classification of Mars terrain images** into five geological classes.

**Final score: 0.73 mean IoU on the test set.**

## The task

We're given **2,615 grayscale images** of the Martian surface, 64×128 pixels
each, together with their ground-truth segmentation masks. Every pixel is
labelled as one of five classes:

| Class | Name | Description |
|-------|------|-------------|
| 0 | Background | Out-of-image pixels, sky, rover hardware, unclassified |
| 1 | Soil | Loose surface material |
| 2 | Bedrock | Exposed solid rock surface |
| 3 | Sand | Fine granular material |
| 4 | Big Rock | Distinct rocky boulders |

The goal is to train a network that takes an image and outputs a 5-class mask
at the same resolution. Evaluation is done with **mean IoU computed over
classes 1–4 only** (background is excluded), which guides several of our
design choices below.

The competition was hosted on CodaLab and ran for ~2 weeks.

## What's in the repo

Two Jupyter notebooks, written for Google Colab with a GPU runtime:

| Notebook | Contents |
|---|---|
| `Deepmindset_Challenge2_model_1.ipynb` | Baseline U-Net + the full data-prep pipeline (outlier removal, label refinement, augmentation, class weights). This is what got us most of the way. |
| `Deepmindset_Challenge2_PCA_model_2.ipynb` | Second model: same encoder/decoder backbone but with an **Atrous Spatial Pyramid Pooling (ASPP)** block at the bottleneck. Used for ensembling with model 1. |

Both notebooks are self-contained: load data → preprocess → train → save
predictions. The final submission averages the per-class probabilities of
the two models and takes argmax.

## How to run it

1. Open the notebooks in Google Colab (or local Jupyter with a CUDA GPU).
2. The first cells expect the dataset in a Google Drive folder mounted at
   `/content/drive/MyDrive/`. Adjust the path constants at the top if your
   layout is different.
3. Run all cells top to bottom. Each notebook saves a Keras model file and
   a CSV of test-set predictions ready for the CodaLab submission.

Required packages (all pre-installed on Colab): TensorFlow/Keras, NumPy,
scikit-learn, matplotlib.

## Pipeline summary

### 1 · Data cleaning

- **Outlier detection** with PCA + Mahalanobis distance flagged a cluster of
  ~110 images containing alien artefacts (all sharing the same broken
  mask). These were removed, bringing the working set to **2,505 images**.
- **Manual label refinement** on ~75 images where the ground-truth mask was
  visibly wrong, with particular focus on the under-represented Big Rock
  class. Polygonal regions of interest were drawn in MATLAB using
  `roipoly()` and merged back into the masks.
- **Class imbalance** for Big Rock was further addressed by duplicating
  the 30 corrected Big-Rock images several times with light geometric
  augmentation.

### 2 · Augmentation

A deliberately light pipeline — the model proved sensitive to heavy
augmentation. Only **random horizontal and vertical flips** survived
testing (with the same transform applied consistently to image and mask).
Contrast, zoom, and brightness adjustments all hurt validation performance.

### 3 · Architecture (Model 1)

A standard **U-Net**:

- 3 encoder blocks (Conv 3×3 → BN → ReLU, twice) with He-Normal init, each
  followed by 2×2 max-pool.
- Bottleneck with 256 filters and L2 regularisation.
- Decoder mirroring the encoder, with skip connections from the
  corresponding encoder block.
- 1×1 conv → softmax output for 5-class pixel probabilities.

**Upsampling choice.** Instead of the usual transposed convolution, we
follow the Distill article on
[deconvolution checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/)
and use **nearest-neighbor upsampling followed by a standard 3×3
convolution**. This produced visibly cleaner masks at the rock boundaries.

### 4 · Training

- Optimiser: **Adam** (SGD and AdamW were worse on this problem).
- Loss: **categorical focal cross-entropy with per-class weights**. Focal loss
  helps with the heavy class imbalance, and the weighting amplifies it
  further on Big Rock.
- Callbacks: `EarlyStopping` and `ReduceLROnPlateau` — smaller learning
  rates near convergence consistently improved validation IoU.

**The single biggest gain came from the class weights.** Setting the
weight for class 0 (background) to **zero** removed a lot of noise from
the loss: background pixels are full of unclassified clutter (rover
hardware, mislabelled terrain) and the evaluation metric ignores them
anyway. After a search, the final weights were

```
[0, 4, 5, 5.5, 50]   # background, soil, bedrock, sand, big rock
```

This single change moved mean IoU from **0.49 → 0.65** on validation.

### 5 · Second model + ensembling (Model 2)

Same U-Net skeleton, but the bottleneck is replaced by an **Atrous
Spatial Pyramid Pooling (ASPP)** block: parallel atrous (dilated)
convolutions with different dilation rates plus a global-average-pooling
branch, all concatenated. This widens the bottleneck's effective
receptive field, which intuitively helps the network see whole rocks
rather than fragments.

The two models are combined by **summing their per-class softmax
probabilities and taking argmax** — a lightweight stacking-style
ensemble, no meta-learner. This bumped the final score from ~0.65 to
**0.727** on the test set.

### 6 · What didn't work

- **Atrous convolutions outside the bottleneck** — the network became
  unstable.
- **Edge maps (Sobel, Canny) and watershed segmentation** as auxiliary
  inputs — the images were too noisy for those algorithms to add signal.
- **Test-time augmentation** — flips at inference didn't meaningfully
  change predictions.

## Limitations and next steps

- **Label noise is the main bottleneck.** A more accurate ground truth
  would let the network learn finer features; we only had time to
  hand-fix ~75 of 2,505 images.
- **Dataset size** is small for a from-scratch segmentation network. Pre-
  training on a related dataset would likely help, but matching the very
  specific Mars-grayscale domain is hard.
- **Class 0 (background)** is heterogeneous and could be split further
  with k-means / clustering, possibly turning some of its pixels into
  usable training signal for the four geological classes.

## Authors

Giacomo Giovanni Papa, Marta Zecchini, Leonardo Salvucci, Alessandro Torazzi —
*Team Deepmindset*, A.Y. 2024/25.
