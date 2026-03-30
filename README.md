# VG-CBM: Visually Grounded Concept Bottleneck Model

Sparse Autoencoder (SAE) + Concept Bottleneck Model on frozen ResNet-50 with SLIC superpixel grounding.
Each prediction is spatially explainable — which concept activated, and where in the image.

---

## Setup

```bash
git clone <this repo>
cd vg_cbm
pip install -r requirements.txt
```

Requirements: Python 3.9+, PyTorch 2.0+. A CUDA GPU is recommended but not required.

---

## Quick Start

### Flowers102

Flowers102 downloads automatically on first run. No manual setup needed.

```bash
python train.py \
    --dataset flowers102 \
    --data_root ./data \
    --run_dir ./runs/flowers \
    --expansion 4 --topk 256 --fg_ratio 0.50 \
    --n_segments 80 --pool_mode max \
    --sae_epochs 30 --head_epochs 40 --stage2_start 10 \
    --w_sp_orth 0.20 --w_div 0.02 --w_l1 0.05 --head_l1 0.001 \
    --batch_size 64 --num_workers 0
```

### Car-Best

Car-Best requires downloading Stanford Cars 196 from Hugging Face and filtering it.
Run the preparation script once before training:

```bash
# Step 1: prepare the dataset (~1.8 GB download, ~5-10 min)
python data/prepare_carbest.py --data_root ./data

# Step 2: train
python train.py \
    --dataset car_best \
    --data_root ./data \
    --run_dir ./runs/car \
    --expansion 4 --topk 256 --fg_ratio 0.30 \
    --n_segments 80 --pool_mode max \
    --sae_epochs 30 --head_epochs 40 --stage2_start 10 \
    --w_sp_orth 0.20 --w_div 0.02 --w_l1 0.05 --head_l1 0.001 \
    --batch_size 64 --num_workers 0
```

Training takes approximately 1–2 hours per dataset on a single GPU.
Outputs are saved to `./runs/{flowers,car}/`:
- `sae_checkpoint.pt` — trained SAE weights
- `head_best.pt` — best CBMHead weights
- `config.json` — full hyperparameter record
- `final_stats.json` — accuracy, K_0.95, intervention results
- `concept_gallery/` — top-30 concept visualizations
- `visuals/final/` — visual strips for test images

### Generating Figures (after training)

```bash
# Figures A-E: pipeline visualization (SLIC, deviation map, FG mask, masking effect, concept maps)
python paper_scripts/compose_figures.py --run_dir ./runs/car     --tag _car     --n 4 --n_e 6
python paper_scripts/compose_figures.py --run_dir ./runs/flowers --tag _flowers --n 4 --n_e 6

# Figure F: progressive concept removal curve
python paper_scripts/gen_fig_F_progressive.py
# -> figures/fig_F_progressive_{car,flowers}.png

# Figure G: CBM reasoning (concept heatmaps + contribution bars)
python paper_scripts/gen_fig_G_results.py
# -> figures/fig_G_reasoning_{car,flowers}.png

# Figure H: concept sparsity curve (K_0.95)
python paper_scripts/gen_fig_H_sparsity.py
# -> figures/fig_H_sparsity.png
```

Pre-generated figures are available in `figures/`.

---

## Dataset Details

### Car-Best

Stanford Cars 196 contains many year-variant classes that are visually indistinguishable
(e.g. "Audi A5 Coupe 2011" vs "Audi A5 Coupe 2012"). Including them inflates class count
without adding separable structure.

Car-Best is constructed from Stanford Cars 196 by:
1. Merging year variants of the same Make+Model+BodyType into one class
2. Dropping classes where a frozen ResNet-50 linear probe scores below 40% per-class accuracy

Result: 31 classes, train 1,345 images / test 1,327 images.

The goal for Car-Best is to test whether the model captures semantically meaningful visual
features distinguishing vehicle categories — body style, proportions, brand-specific elements.

The 31 classes:

```
 0  AM General Hummer SUV 2000
 1  Aston Martin V8 Vantage Conv. 2012
 2  Audi S5 Convertible 2012
 3  Bentley Continental Supersports
 4  Buick Rainier SUV 2007
 5  Chevrolet Corvette ZR1 2012
 6  Chrysler PT Cruiser Conv. 2008
 7  Dodge Caravan Minivan 1997
 8  Dodge Ram Pickup 3500 Quad Cab
 9  Dodge Sprinter Cargo Van 2009
10  Ferrari 458 Italia Conv. 2012
11  Ford Freestar Minivan 2007
12  Ford F-150 Regular Cab 2007
13  Ford E-Series Wagon Van 2012
14  GMC Savana Van 2012
15  Geo Metro Convertible 1993
16  HUMMER H2 SUT Crew Cab 2009
17  Honda Odyssey Minivan 2012
18  Hyundai Tucson SUV 2012
19  Jaguar XK XKR 2012
20  Jeep Liberty SUV 2012
21  Lamborghini Gallardo LP 570-4
22  Land Rover Range Rover SUV 2012
23  MINI Cooper Roadster Conv. 2012
24  Mercedes-Benz 300-Class Conv. 1993
25  Mercedes-Benz Sprinter Van 2012
26  Nissan NV Passenger Van 2012
27  Ram C/V Cargo Van Minivan 2012
28  Spyker C8 Convertible 2009
29  Toyota 4Runner SUV 2012
30  smart fortwo Convertible 2012
```

### Flowers102

Oxford Flowers102 is used to evaluate whether the model correctly identifies the
discriminative visual parts of each flower species. Flowers vary in petal shape,
color distribution, stamen structure, and leaf arrangement — spatially localized
features that require the model to attend to the right regions.
The concept attribution maps and intervention results show whether the model's active
concepts correspond to the relevant floral parts rather than global color or texture.

Flowers102 is downloaded automatically via torchvision on first run.

---

## Model Overview

### Pipeline

```
Input Image (224x224)
    |
Frozen ResNet-50 (layer3 tap) -> (B, 1024, 14, 14)
    |
SLIC Superpixels (n_segments=80) -> 80 superpixel regions
    |
Superpixel Feature Pooling -> (B, 80, 1024)
    |
Foreground Masking (top fg_ratio % by deviation from mean) -> background suppressed
    |
Sparse SAE (TopK=256, K=4096) -> (B, 80, 4096)  — 256 concepts active per superpixel
    |
fg_z_pool (MAX over superpixels) -> (B, 4096)   — image-level concept representation
    |
Non-negative CBMHead -> class logits
```

### Design Choices

TopK SAE activates exactly 256 concepts per superpixel, producing sparse and interpretable codes.

The non-negative CBMHead applies F.relu(weight) so concepts can only positively support a
prediction — never suppress it. This ensures greedy concept removal is monotone:
removing the top-contributing concept always decreases (or maintains) confidence.
This property makes it possible to directly measure which concepts sustain a prediction (Figure F).

Foreground masking retains only the top fg_ratio % of superpixels by deviation from the image
mean, filtering background noise before passing to the SAE.

MAX pooling captures the strongest concept activation anywhere in the foreground region.
This works better than mean pooling for localized features like car grilles or flower stamens.

---

## Why Layer3

The analysis in `analysis/` compares ResNet-50 layers on their ability to support
SAE-based classification.

### Layer Comparison (CIFAR-100, expansion=8)

| Layer  | d_in | Spatial | Raw Linear Probe | SAE-Z Acc | Delta   |
|--------|------|---------|-----------------|-----------|---------|
| layer2 | 512  | 28x28   | 32.6%           | 52.8%     | +20.2%  |
| layer3 | 1024 | 14x14   | 61.6%           | 75.0%     | +13.4%  |
| layer4 | 2048 | 7x7     | 72.3%           | 74.9%     | +2.6%   |

SAE-Z Acc is the accuracy of a free linear probe trained on z_pool (the SAE output features).

Layer4 has the highest raw linear probe score, but the SAE barely improves it (+2.6%).
Layer4 features are already saturated for linear classification — there is little latent
structure left for the SAE to decompose into interpretable concepts.
Layer3 still has decomposable structure, giving the SAE room to improve (+13.4%) and
reaching the highest SAE-Z Acc overall.
Layer3 also has 14x14 spatial resolution, which is necessary for meaningful superpixel-level
localization. Layer4's 7x7 is too coarse for part-level concept grounding.

To reproduce this analysis:
```bash
python analysis/layer_scan.py --backbone resnet50 --expansion 8 --l1_lambda 2.37e-2
python analysis/plot_layer_comparison.py
# results saved to analysis/results/
```

---

## Results

| Dataset    | Our Model | Linear Probe | Gap   | Config                      |
|------------|-----------|-------------|-------|-----------------------------|
| Car-Best   | 73.7%     | 78.3%       | -4.6% | topk=256, fg=0.30, pool=max |
| Flowers102 | 82.9%     | 83.9%       | -1.0% | topk=256, fg=0.50, pool=max |

The gap to the linear probe is the cost of interpretability: TopK sparsity and the
non-negative head both constrain the representation.
Flowers102 closes this gap to -1.0%, suggesting the model's superpixel-level concepts
align well with the discriminative floral parts.

### Concept Intervention (Figure F)

Greedy removal of top-8 concepts, confidence before and after:

Car-Best:
- Chrysler PT Cruiser Conv.:  93.2% -> 21.4%  (drop: 71.8%)
- Mercedes-Benz 300-Class:    88.3% -> 16.0%  (drop: 72.3%)
- Jaguar XK XKR:              80.8% ->  9.5%  (drop: 71.3%)

Flowers102:
- Class 24:  91.0% -> 23.4%  (drop: 67.5%)
- Class  3:  86.5% -> 13.8%  (drop: 72.7%)
- Class 34:  81.7% ->  7.4%  (drop: 74.4%)

### K_0.95

K_0.95 is the number of concepts needed to cover 95% of the prediction confidence mass
(computed per correctly-predicted test sample, then averaged).
A lower K_0.95 means the prediction is concentrated in fewer concepts.

- Car-Best:   K_0.95 = 653  (15.9% of 4096)
- Flowers102: K_0.95 = 614  (15.0% of 4096)

95% of prediction confidence is explained by roughly 15% of the concept space,
demonstrating that the model's decisions are driven by a sparse, interpretable subset
of the learned concept dictionary.

---

## SAE Diagnostic

`analysis/diagnose_sae_z.py` decomposes the accuracy gap into two sources.

```
[A] GAP (raw layer4) -> free linear probe:   78.2%   linear probe baseline
[B] SAE-Z -> free linear probe:              72.6%   SAE feature quality
[C] SAE-Z -> non-negative probe:             63.6%   with CBMHead constraint
[D] Actual CBM-SAE (mean pool):              66.5%
```

SAE features lose ~5.6% vs raw (A to B). The non-negative constraint costs another ~9% (B to C).
Both are expected: the SAE trades some classification capacity for a decomposed, sparse
representation, and the non-negative head trades further capacity for monotone interventions.
Switching to MAX pooling brings the final model to 73.7%, recovering most of the gap.

To run the diagnostic on the trained car model:
```bash
python analysis/diagnose_sae_z.py --run_dir ./models/car --data_root ./data
```

---

## Hyperparameter Search

| Change                   | Effect on Car | Note                                         |
|--------------------------|---------------|----------------------------------------------|
| topk: 30 -> 256          | +12%          | relieved SAE bottleneck                      |
| pool: mean -> max        | +7%           | captures localized features (grille, badge)  |
| fg_ratio: 0.30 (car)     | optimal       | higher fg_ratio increases background noise   |
| fg_ratio: 0.50 (flowers) | optimal       | flowers need broader spatial coverage        |
| n_segments: 80           | optimal       | increasing to 160 gave no improvement        |
| topk: 256 vs 512         | 256 better    | very high topk degrades SAE training quality |

---

## Generating Paper Figures

### Figures A-E (pipeline visualization)

These are generated from the visual strips in `models/{car,flowers}/visuals/final/`.
Run after training is complete.

```bash
# from the vg_cbm/ directory
python paper_scripts/compose_figures.py --run_dir ./models/car     --tag _car     --n 4 --n_e 6
python paper_scripts/compose_figures.py --run_dir ./models/flowers --tag _flowers --n 4 --n_e 6
# figures saved to ./figures/
```

### Figure F (progressive concept removal)

```bash
python paper_scripts/gen_fig_F_progressive.py
# -> figures/fig_F_progressive_car.png
# -> figures/fig_F_progressive_flowers.png
```

---

## Directory Structure

```
vg_cbm/
├── train.py                        main training script
├── data/
│   └── prepare_carbest.py          downloads and filters Car-Best from Stanford Cars 196
├── src/
│   ├── backbone.py                 ResNet/ViT feature extractor + forward hook
│   ├── models.py                   FeatureNorm, SparseSAE, CBMHead, fg_z_pool
│   ├── slic_utils.py               SLIC computation, superpixel feature extraction
│   ├── training.py                 SAE and head training loops
│   ├── evaluate.py                 K_0.95 and intervention evaluation
│   ├── losses.py                   sp_orth and diversity losses
│   ├── visualize.py                visual strip generation
│   └── datasets.py                 dataset loading utilities
├── models/
│   ├── car/                        Car-Best final model
│   │   ├── sae_checkpoint.pt
│   │   ├── head_best.pt
│   │   ├── config.json
│   │   ├── final_stats.json
│   │   ├── carbest_classes.json
│   │   ├── concept_gallery/        top-30 concept visualizations
│   │   └── visuals/final/          post-training visual strips
│   └── flowers/                    Flowers102 final model
│       ├── sae_checkpoint.pt
│       ├── head_best.pt
│       ├── config.json
│       ├── final_stats.json
│       ├── concept_gallery/
│       └── visuals/final/
├── figures/                        paper figures (A-H)
│   ├── fig_A_slic_{car,flowers}.png
│   ├── fig_B_deviation_{car,flowers}.png
│   ├── fig_C_fg_mask_{car,flowers}.png
│   ├── fig_D_masking_{car,flowers}.png
│   ├── fig_E_concepts_{car,flowers}.png
│   ├── fig_F_progressive_{car,flowers}.png
│   ├── fig_G_reasoning_{car,flowers}.png
│   └── fig_H_sparsity.png
├── paper_scripts/
│   ├── compose_figures.py          generates figures A-E
│   ├── gen_fig_F_progressive.py    generates figure F
│   ├── gen_fig_G_results.py        generates figure G (CBM reasoning)
│   └── gen_fig_H_sparsity.py       generates figure H (concept sparsity K_0.95)
├── analysis/
│   ├── layer_scan.py               layer2/3/4 SAE-Z accuracy comparison (CIFAR-100)
│   ├── plot_layer_comparison.py    plots the layer comparison
│   ├── diagnose_sae_z.py           SAE quality diagnostic
│   └── results/
│       ├── layer_comparison.csv
│       └── plots/
├── requirements.txt
└── .gitignore
```
