# Physics-Aware Multi-Task Learning for Forest Biomass Estimation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SAR](https://img.shields.io/badge/Data-Sentinel--1-orange.svg)](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)

## Overview

This project develops a **multi-task deep learning framework** for estimating forest structural attributes from Sentinel-1 SAR imagery. Using the synthetic I-MAESTRO dataset spanning three European forest regions, we jointly predict **genus classification**, **canopy height**, and **aboveground biomass** through a shared encoder architecture with physics-based constraints.

*** *This repository contains the physics-aware deep learning part for the Master's thesis*

### Key Features

- **Multi-task U-Net architecture** with shared encoder and task-specific decoders
- **Uncertainty-weighted loss** for automatic task balancing
- **Allometric constraint** enforcing ecological relationships between height and biomass
- **Gradient analysis** including gradient alignment and feature similarity (CKA)

## Mission & Objectives

### Scientific Goals

Forest monitoring at scale requires accurate estimation of structural attributes for carbon accounting, biodiversity assessment, and sustainable management. This project addresses the challenge of extracting multiple forest properties from freely available SAR data.

**Primary objectives:**
- Develop multi-task learning models that leverage shared representations across related forest attributes
- Incorporate ecological knowledge (allometric scaling) as physics-based constraints
- Evaluate whether joint training improves generalization compared to single-task baselines
- Analyze task relationships through gradient alignment and representation similarity

### Technical Challenges

1. **SAR signal saturation** - C-band backscatter saturates at ~50-100 t/ha, limiting biomass estimation in mature forests
2. **Class imbalance** - Dominant genera (Fagus, Abies, Pinus) vastly outnumber rare species
3. **Spatial autocorrelation** - Requires careful data splitting to prevent inflated performance estimates
4. **Task competition** - Balancing competing objectives in multi-task optimization

## Dataset

### I-MAESTRO Synthetic Forest Landscapes

The project uses three European forest sites with synthetic tree-level inventory data:

| Site | Location | Area | Trees | Dominant Genera |
|------|----------|------|-------|-----------------|
| **Bauges** | France | 64,000 ha | 10.5M | Fagus, Abies, Pinus |
| **Milicz** | Poland | 20,000 ha | 4.6M | Pinus, Quercus |
| **Sneznik** | Slovenia | 5,300 ha | 0.8M | Fagus, Abies |

**Input features:** Sentinel-1 GRD dual-polarization (VV, VH) backscatter  
**Target variables:** Dominant genus (6 classes), canopy height (height95), biomass (t/ha)  
**Spatial resolution:** 25m grid  
**Training patches:** 724 (64×64 pixels) | Validation: 145 | Test: 109

## Results

### Performance Summary

| Task | Metric | Single-Task | Multi-Task | Change |
|------|--------|-------------|------------|--------|
| **Genus Segmentation** | Mean Dice | 0.366 | 0.366 | 0.0% |
| **Height Regression** | R² | 0.255 | 0.265 | +3.7% |
| **Biomass Regression** | R² | 0.024 | 0.039 | - |

### Key Findings

✅ **Multi-task learning achieves comparable performance** to single-task baselines while using a single shared encoder  
✅ **Height-biomass tasks show strong gradient alignment** (cosine similarity 0.4-0.9), validating allometric coupling  
✅ **CKA analysis reveals task hierarchy**: height-biomass (most similar) > segmentation-height > segmentation-biomass  
⚠️ **C-band SAR limitations** constrain absolute performance, particularly for high biomass estimation (saturation effects)

### Visualization Examples

![Multi-task predictions](@plots/training-results/mtl/mtl-all-tasks-samples.png)
*Example predictions showing VV/VH inputs and genus/biomass/height outputs*

## Repository Structure

```
├── @data-preprocessing/     # Data pipeline scripts
├── @training/              # Model training and experiments
├── @plots/                 # Visualization outputs
│   ├── data-insight/       # Data exploration plots
│   └── training-results/   # Model performance plots
└── @data/                  # Raw and processed datasets
```

## Quick Start

### Prerequisites

```bash
python >= 3.8
pytorch >= 2.0
rasterio
numpy
scikit-learn
```

### Data Preprocessing

See [`@data-preprocessing/README.md`](@data-preprocessing/README.md) for detailed pipeline documentation.

```bash
cd @data-preprocessing
python main.py --sites bauges milicz sneznik
```

### Model Training

See [`@training/README.md`](@training/README.md) for training configurations and experiments.

```bash
cd @training
python train_multitask.py --config configs/mtl_uncertainty.yaml
```

## Methodology Highlights

### Architecture

- **Shared encoder**: 4-stage U-Net (128→256→512→1024 channels) with strided convolutions
- **Task-specific decoders**: Independent upsampling paths with skip connections
- **Dropout regularization**: 0.4 (segmentation), 0.2 (regression)

### Training Strategy

- **Uncertainty weighting**: Automatic task balancing via learned homoscedastic uncertainty
- **Allometric constraint**: Regularization enforcing B = exp(α) × H^β relationship
- **Spatial blocking**: 4×4 patch blocks ensure geographic separation of train/val/test sets
- **Early stopping**: Patience=10 epochs monitoring validation metrics

### Analysis Tools

- **Gradient alignment**: Cosine similarity between task-specific gradients in shared encoder
- **CKA similarity**: Centered Kernel Alignment measuring decoder representation similarity
- **Layer-wise visualization**: Feature map analysis across encoder/decoder depths

## Conclusions

This work demonstrates that **multi-task learning with physics-based constraints** provides a viable approach for forest attribute estimation from SAR data. While absolute performance is limited by C-band saturation, the framework successfully:

1. **Learns shared representations** beneficial across multiple forest attributes
2. **Automatically balances competing tasks** through uncertainty weighting
3. **Incorporates ecological knowledge** via allometric regularization
4. **Provides interpretable analysis** of task relationships and feature learning

**Future directions** include integrating longer wavelength SAR (L-band), multi-temporal features, and optical data fusion to overcome current limitations.

---

## Contributing & Feedback

⭐ **If you find this work useful, please consider starring the repository!** Feel free to open an issue or start a discussion. Feedback and suggestions are always welcome.
