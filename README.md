# SAHG: Sector-Anisotropic Hyperbolic Graph model for Social Bot Detection

This repository contains the official implementation of **SAHG**, a dual-channel framework that combines hyperbolic geometry, sector-aware prototype learning, and GraphSAGE message passing for social bot detection.

## Architecture

SAHG encodes each user's feature vector into a **Sector-Anisotropic Hyperbolic (SAH) space**, extracting three geometric signals:
- **r** — radial coordinate (anomaly intensity)
- **H** — sector entropy (bots concentrate in fewer directions; humans spread broadly)
- **A** — maximum sector alignment score

A GraphSAGE channel augments the original-channel features with 2-hop neighborhood aggregation. Both channels' geometric signals are concatenated and fed to a lightweight classification head.

## Repository Structure

```
SAHG/
├── eval.py                  # Main evaluation entry point
├── requirements.txt
├── models/
│   └── sah.py               # SAHEncoder, LocalWarpNet, SectorPrototypes
├── algorithm/
│   └── sahg_model.py        # SAHGSingleDetector, FocalBCE, train_single_graph
└── data/
    ├── loader.py            # Dataset loaders for MGTAB, Fox8-23, BotSim-24
    ├── preprocess_fox8.py   # Fox8-23 preprocessing (ndjson → CSV)
    └── preprocess_botsim.py # BotSim-24 preprocessing (CSV+JSON → CSV)
```

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: 
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- scikit-learn ≥ 1.3

## Data Preparation

**MGTAB**: Download from the [MGTAB repository](https://github.com/GraphDetec/MGTAB) and place under `data/mgtab/`.

**Fox8-23**: Download the ndjson file and run:
```bash
python data/preprocess_fox8.py \
    --ndjson /path/to/fox8_23_dataset.ndjson \
    --out_dir data/fox8_23/
```

**BotSim-24**: Download CSV and JSON files and run:
```bash
python data/preprocess_botsim.py \
    --csv  /path/to/BotSim-24_user.csv \
    --json /path/to/BotSim-24_user_post_comment.json \
    --out_dir data/botsim_24/
```

## Quick Start (Evaluation)

You can run the model evaluation using the provided `eval.py` script. The script automatically handles data loading, model initialization, training, and testing.

```bash
# Evaluate on all three datasets (MGTAB, Fox8-23, BotSim-24) using seeds 0, 1, and 2:
python eval.py \
    --mgtab_data  data/mgtab \
    --fox8_data   data/fox8_23 \
    --botsim_data data/botsim_24

# Evaluate on a single dataset (e.g., only MGTAB):
python eval.py --skip_fox8 --skip_botsim

# Run evaluation with custom random seeds:
python eval.py --mgtab_seeds 0 1 2
```