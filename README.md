# M4: Multi-Proxy Multi-Gate Mixture of Experts Network for Multiple Instance Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-%23EE4C2C.svg)](https://pytorch.org/)

Official implementation of **M4: Multi-Proxy Multi-Gate Mixture of Experts Network for Multiple Instance Learning in Histopathology Image Analysis**

## Abstract
Multiple instance learning (MIL) has been successfully applied for whole slide images (WSIs) analysis in computational pathology, enabling various prediction tasks from tumor subtyping to inferring genetic mutations and multi-omics biomarkers. However, existing MIL methods predominantly focus on single-task learning, resulting in both low efficiency and oversight of inter-task relatedness. 

To address these limitations, we propose M4 - a novel architecture featuring:
1. **Multi-gate Mixture-of-Experts** for simultaneous prediction of multiple genetic mutations from a single WSI
2. **Multi-proxy AMIL construction** to effectively capture patch-patch interactions within WSIs

Our model demonstrates significant improvements across 5 TCGA datasets compared to state-of-the-art single-task methods.

## Environment Setup
- **PyTorch 2.0.1**

##Dataset Preparation
- Pretrained WSI feature extractor: [RetCCL](https://github.com/Xiyue-Wang/RetCCL)
- Genetic mutation features are provided in `M4_csv/` directory:
  - `10genes/`: Multi-task dataset
  - `1gene/`: Single-task dataset

## Quick Start
1. Clone repository:
```bash
git clone [repository_url]
cd M4
```
2. Run main script:
```bash
sh scripts/M4-main.sh
```
## Repository Structure
├── M4_csv/                 # Genetic mutation data
│   ├── 10genes/           # Multi-task dataset
│   └── 1gene/             # Single-task dataset
├── models/                #models
│   ├── M4.py   
│   └── MMoE.py
│   └── AMIL.py
├── scripts/               # Execution scripts
│   └── M4-main.sh   
│   └── MMoE.sh
│   └── AMIL.sh
├── main.py
├── Dataset.py
├── scheduler.py
├── train_val_multi.py
