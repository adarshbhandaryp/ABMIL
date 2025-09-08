# ABMIL
Robust Breast MRI Classification via Multi-Sequence Attention-Based Multiple Instance Learning and Focal Loss Optimization

We propose a multi-sequence Attention-Based Multiple Instance Learning (ABMIL) framework for subject-level breast MRI classification. Each breast volume is represented by 32 slices across pre-contrast, post-contrast, and subtraction sequences. Slice-level features are extracted using a pretrained Swin Transformer Tiny backbone and aggregated with a gated attention mechanism to form patient-level embeddings. To address severe class imbalance, we adopt focal loss with γ=5.0, which proved optimal in enhancing sensitivity to malignant cases. Our approach leverages full 3D volumes without lesion annotations, integrating temporal contrast dynamics to achieve robust and clinically meaningful classification across heterogeneous institutions

Code for ODELIA Breast MRI Challenge 2025



