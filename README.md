# Robust Breast MRI Classification via Multi-Sequence Attention-Based Multiple Instance Learning and Focal Loss Optimization
This repository contains code for **Attention-Based Multiple Instance Learning (ABMIL)** models for **breast MRI classification**.  

# 🩺 Code for ODELIA Breast MRI Challenge 2025
## Overview  
The **ODELIA Breast MRI Challenge 2025** benchmarks subject-level classification on a multi-center dataset of unilateral breast MRI in NIfTI format with three labels: **normal, benign, and malignant**. The goal is to advance clinically meaningful AI by tackling class imbalance, temporal contrast dynamics, and multi-sequence integration. Participants build models that ingest pre- and post-contrast sequences (and subtraction) and produce patient-level predictions evaluated on robust metrics. The challenge emphasizes reproducibility and fairness across institutions, encouraging architectures that learn from full 3D volumes without lesion annotations. Results aim to accelerate real-world deployment of reliable screening and diagnostic support systems across heterogeneous clinical settings.

## Method  
We present a multi-sequence **Attention-Based Multiple Instance Learning (ABMIL)** framework for subject-level breast MRI classification. Each breast volume is represented by 32 axial slices with **pre-contrast, early post-contrast, and on-the-fly subtraction** channels. Slice features are extracted with **Swin Transformer Tiny** (ImageNet-pretrained) and aggregated via **gated attention** to form a patient-level embedding. To mitigate pronounced class imbalance, we optimize with **focal loss** and found **γ = 5.0** consistently strongest across validation and pre-evaluation tests. The approach operates without lesion annotations, leverages temporal enhancement patterns, and delivers robust performance across institutions—tailored for scalable, clinically relevant evaluation.

# Code


The training pipeline is controlled by **CSV configuration files** located in the [`config/`](config/) folder.

---

## 🚀 Running the Code

```bash
pip install -r requirements.txt

python main.py \
  --configuration_file "config/config_mil_cross_nii.csv" \
  --project_name "Breast MRI MIL Cross Attention"
configuration_file → Path to one of the CSV configs in config/.
project_name → Custom name for logging runs in Weights & Biases
```

## Contact Us  
- **Dr. Tri-Thien Nguyen** – [tri-thien.nguyen@fau.de](mailto:tri-thien.nguyen@fau.de) · [Profile](https://lme.tf.fau.de/person/ttnguyen)  
- **Adarsh Bhandary Panambur** – [adarsh.bhandary.panambur@fau.de](mailto:adarsh.bhandary.panambur@fau.de) · [Profile](https://lme.tf.fau.de/person/panambur/)  
- **Pattern Recognition Lab (FAU Erlangen-Nürnberg)** – [https://lme.tf.fau.de/](https://lme.tf.fau.de/)
