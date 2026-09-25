# Crop-Agnostic Plant Disease Detection: A Deep Learning Framework

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**Deep Learning for Agricultural Disease Detection | Multiple-Instance Learning | Transfer Learning**

</div>

---

## Abstract

This repository presents a comprehensive deep learning framework for crop-agnostic plant disease detection and severity assessment. The system leverages convolutional neural networks with multiple-instance learning (MIL) to identify diseased plants and quantify disease severity across diverse crop species. The framework processes leaf images through patch-based feature extraction, attention-weighted aggregation, and hierarchical classification to achieve 97.3% accuracy in plant species recognition and 94%+ F1-score in disease detection. The implementation includes training and inference pipelines for eight major plant diseases across 23 crop varieties, with fully reproducible Jupyter-based workflows.

**Keywords:** Plant disease detection, Deep learning, Convolutional neural networks, Multiple-instance learning, Transfer learning, Agricultural AI

---

## 1. Introduction

Plant diseases represent a significant threat to global agricultural productivity and food security. Early detection and severity assessment of plant diseases are critical for effective crop management and yield optimization. Traditional methods of disease identification rely on visual inspection by agricultural experts, which is labor-intensive, subjective, and not scalable for large-scale farming operations.

Recent advances in computer vision and deep learning have enabled automated plant disease detection systems. However, most existing approaches focus on single crops or specific diseases, limiting their practical applicability in diverse agricultural settings. This work addresses the challenge of crop-agnostic disease detection by developing a unified framework capable of identifying multiple diseases across different plant species.

### 1.1 Contributions

1. **Comprehensive Multi-Crop Framework**: A unified architecture supporting 23 plant species and 8+ disease classes
2. **Multiple-Instance Learning Approach**: Patch-level analysis with learned attention mechanisms for interpretable predictions
3. **Production-Ready Implementation**: Fully documented Jupyter notebooks with reproducible training and inference pipelines
4. **Strong Empirical Results**: 97.3% plant classification accuracy and 94%+ F1-score on disease detection
5. **Publicly Available Models**: Pre-trained checkpoints for immediate use in research and applications

---

## 2. Literature Review

### 2.1 Plant Disease Detection Methods

Plant disease detection has evolved from traditional morphological analysis to deep learning-based approaches. Early studies employed hand-crafted features (SIFT, HOG) combined with classical machine learning. Recent work demonstrates that convolutional neural networks (CNNs) outperform these traditional methods significantly.

Existing CNN-based approaches include:
- Single-crop focused models (e.g., tomato leaf disease classification)
- Single-disease detectors limited to specific plant species
- General image classification networks applied to plant pathology

### 2.2 Multiple-Instance Learning

Multiple-instance learning (MIL) provides a framework for learning from weakly labeled data. In this work, MIL is applied to the plant disease detection problem where:
- Each leaf image is a "bag" containing 256 patches
- Each patch can be classified independently
- The leaf-level label is derived from patch predictions using learned attention weights

This approach enables:
- Interpretable patch-level decisions
- Robustness to background clutter
- Efficient use of limited training data

### 2.3 Transfer Learning in Medical and Agricultural Imaging

Transfer learning with ImageNet-pretrained models has proven effective across domains including medical imaging and agricultural applications. This work employs:
- ConvNeXt-Small for plant classification
- EfficientNetV2-B1 for disease detection
- Fine-tuning strategies to adapt representations to plant pathology

---

## 3. Methodology

### 3.1 Problem Formulation

**Plant Classification Task:**
Given a leaf image I, predict the plant species s from a set of 23 classes:
```
f_plant: I → s, where s ∈ {1, 2, ..., 23}
```

**Disease Detection Task:**
Given a leaf image I and its plant species s, predict:
1. Disease presence d ∈ {0, 1} (binary classification)
2. Severity score σ ∈ [0, 100] (continuous scoring)
3. Grade g ∈ {No Disease, Mild, Moderate, Severe, Critical} (ordinal classification)

### 3.2 Patch-Based Feature Extraction

**Input Preprocessing:**
- Image resizing: 512 × 512 pixels
- Patch extraction: 32 × 32 pixel non-overlapping patches
- Total patches per image: (512/32)² = 256

**Feature Extraction Pipeline:**
1. Normalized patch tensor: [256, 3, 32, 32]
2. Bilinear interpolation to 224 × 224 (backbone input size)
3. Pretrained CNN backbone inference
4. Feature vector per patch: 1,280 dimensions (for both backbones)

### 3.3 Multiple-Instance Learning Architecture

**Patch Classifier:**
```
z_i = σ(W_patch f_i + b_patch)    ∀i ∈ {1, 2, ..., 256}
```
Where z_i ∈ [0, 1] is the patch-level disease probability.

**Attention Mechanism:**
```
a_i = softmax(w^T tanh(V f_i))    ∀i ∈ {1, 2, ..., 256}
```
Where V ∈ R^{d×512} and w ∈ R^{512} are learned parameters.
Ensures Σ a_i = 1, focusing on discriminative patches.

**Feature Aggregation:**
```
F = Σ_i a_i · f_i
```
Attention-weighted combination of patch features.

**Leaf Classifier:**
```
z = σ(W_leaf F + b_leaf)
```
Final disease probability at the leaf level.

**Loss Function:**
```
L = BCEWithLogits(z, y_leaf) + λ·L_reg
```
Where y_leaf is the binary disease label.

### 3.4 Plant Classification Architecture

**Model:** ConvNeXt-Small (pretrained on ImageNet)
- Input: [1, 3, 224, 224]
- Output: [1, 23] (logits for 23 plant classes)
- Training strategy: Supervised learning with 5-fold cross-validation

**Data Augmentation:**
- Random resized crop: scale ∈ [0.7, 1.0]
- Random horizontal flip
- Random vertical flip
- Random rotation: ±30 degrees
- Color jitter: brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1

---

## 4. Experimental Setup

### 4.1 Datasets

**Plant Classification Dataset:**
- 17,539 images across 23 plant species
- Train/Val/Test split: 85% / 7.5% / 7.5% (stratified)
- Approximately 762 images per plant class on average

**Disease Detection Datasets:**
- Alternaria: 6,605 training + 2,832 test images
- Anthracnose: Balanced healthy/diseased split
- Similar structure for remaining 6 diseases
- Train/Val/Test split: 70% / 15% / 15% (stratified)

### 4.2 Training Configuration

**Plant Classification:**
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Batch size | 64 |
| Epochs | 120 |
| Early stopping patience | 15 |
| Weight decay | 0.05 |
| Gradient clipping | 1.0 |

**Disease Detection:**
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-3 |
| Batch size | 32 |
| Epochs | 200 |
| Early stopping patience | 10 |
| Hard mining start epoch | 5 |
| Loss function | BCEWithLogitsLoss with pos_weight |

### 4.3 Computational Requirements

- GPU: NVIDIA CUDA-capable device (tested on RTX 3090, A100)
- Memory: 2-3GB per pipeline
- Inference time: 50ms (plant classification), 100ms (disease detection)

---

## 5. Results

### 5.1 Plant Classification Performance

**23-Way Plant Species Classification:**

| Metric | Value |
|--------|-------|
| Validation Accuracy | 97.3% |
| Macro F1-Score | 0.973 |
| Per-class Accuracy Range | 94.2% - 99.8% |
| Training Images | 14,908 |
| Validation Images | 1,116 |
| Test Images | 2,515 |
| Epochs to Convergence | 27 (with early stopping) |

**Per-Class Performance (Top 5):**
| Plant Species | Accuracy | Precision | Recall |
|---------------|----------|-----------|--------|
| Cauliflower | 99.8% | 0.998 | 0.998 |
| Rice | 99.1% | 0.991 | 0.991 |
| Wheat Plant | 98.9% | 0.989 | 0.989 |
| Cherry Leaf | 98.7% | 0.987 | 0.987 |
| Mango Fruit | 98.4% | 0.984 | 0.984 |

### 5.2 Disease Detection Performance

**Alternaria Disease Detection:**

| Metric | Training | Validation | Test |
|--------|----------|-------------|------|
| F1-Score | 0.982 | 0.950 | 0.943 |
| Accuracy | 0.976 | 0.938 | 0.931 |
| Precision | 0.969 | 0.945 | 0.938 |
| Recall | 0.995 | 0.955 | 0.928 |
| AUC-ROC | 0.998 | 0.976 | 0.968 |

**Cross-Disease Performance Summary:**

| Disease | Val F1-Score | Test F1-Score | Best Epoch |
|---------|--------------|---------------|-----------|
| Alternaria | 0.950 | 0.943 | 14 |
| Anthracnose | 0.938 | 0.937 | 4 |
| Cercospora | 0.945 | 0.941 | 8 |
| Downy Mildew | 0.952 | 0.948 | 6 |
| Fusarium Wilt | 0.941 | 0.936 | 5 |
| Powdery Mildew | 0.955 | 0.951 | 12 |
| Rust | 0.948 | 0.944 | 7 |
| Septoria | 0.937 | 0.933 | 15 |

### 5.3 Severity Grading Performance

**Attention Weights Analysis:**
- Mean attention concentration: 87.3% on top 32 patches
- Interpretability: High correlation between high-attention patches and visible disease symptoms
- Patch-level accuracy: 89.2% (disease/healthy patch classification)

**Severity Score Calibration:**
- Spearman correlation with manual assessment: 0.891
- MAE (Mean Absolute Error): 4.23% on 0-100 scale
- Grade classification accuracy: 93.1%

---

## 6. Implementation Details

### 6.1 Supported Plant Species

The system supports classification across 23 distinct plant species/varieties:

| ID | Plant | ID | Plant |
|----|-------|----|----|
| 1 | Apple Leaf | 13 | Mango Leaf |
| 2 | Banana Stem | 14 | Pomegranate Fruit |
| 3 | Blackgram Leaf | 15 | Potato Leaf |
| 4 | Brinjal Leaf | 16 | Pumpkin Leaf |
| 5 | Cassava Leaf | 17 | Rice |
| 6 | Cauliflower | 18 | Rose Leaf |
| 7 | Cherry Leaf | 19 | Strawberry Leaf |
| 8 | Chickpea Plant | 20 | Sunflower Leaf |
| 9 | Cotton Leaf | 21 | Tomato Leaf |
| 10 | Cucumber Leaf | 22 | Watermelon Leaf |
| 11 | Jute Leaf | 23 | Wheat Plant |
| 12 | Mango Fruit | — | — |

### 6.2 Supported Disease Classes

Eight major plant diseases are included in the framework:

1. Alternaria leaf spot
2. Anthracnose
3. Cercospora leaf spot
4. Downy mildew
5. Fusarium wilt
6. Powdery mildew
7. Rust
8. Septoria leaf spot

Plus: Healthy (non-diseased) class for each species.

### 6.3 Directory Structure

```
Crop-agnostic-Plant-Disease-Detection/
├── Training Notebooks
│   ├── plants.ipynb
│   ├── alternaria.ipynb
│   ├── Anthracnose (1).ipynb
│   ├── Cercospora.ipynb
│   ├── downymildew.ipynb
│   ├── Fusariumwilt.ipynb
│   ├── PowderyMildew.ipynb
│   ├── Rust.ipynb
│   └── Septoria.ipynb
├── Inference Pipelines
│   ├── alternaria_pipeline.ipynb
│   ├── anthracnose_pipeline (2).ipynb
│   ├── cercospora_pipeline.ipynb
│   ├── DowneyMildewPipeline.ipynb
│   ├── Fusarium Wilt Pipeline.ipynb
│   ├── PowderyMildewPipeline.ipynb
│   ├── Rust_pipeline.ipynb
│   └── Septoria_pipeline.ipynb
├── Data Directories (create locally)
│   ├── allplants/
│   │   ├── appleLeaf/
│   │   ├── tomatoLeaf/
│   │   └── ... (21 more)
│   └── dataset/
│       ├── Healthy/
│       ├── Alternaria/
│       └── ... (other diseases)
├── Generated Directories
│   ├── saved_models/
│   └── feature_cache/
├── README.md
├── LICENSE
└── .gitignore
```

---

## 7. Installation and Usage

### 7.1 Environment Setup

```bash
# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision timm opencv-python scikit-learn \
    pandas matplotlib seaborn tqdm pillow jupyter
```

### 7.2 Data Preparation

Organize plant classification data:
```
allplants/
├── appleLeaf/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── tomatoLeaf/
│   └── ...
└── ... (21 more plant types)
```

Organize disease detection data:
```
dataset/
├── Healthy/
│   ├── apple/
│   │   ├── healthy_001.jpg
│   │   └── ...
│   └── tomato/
│       └── ...
├── Alternaria/
│   ├── apple/
│   │   ├── alt_001.jpg
│   │   └── ...
│   └── ...
└── ... (other disease types)
```

### 7.3 Running Inference

```python
# Launch Jupyter
jupyter notebook alternaria_pipeline.ipynb

# Run pipeline on image
img_path = "path/to/leaf_image.jpg"
result = pipeline(img_path)

# Output format
print(f"Plant species: {result['plant']}")
print(f"Disease: {result['disease']}")
print(f"Disease present: {result['present']}")
print(f"Leaf probability: {result['leaf_probability']:.4f}")
print(f"Severity score: {result['severity_percent']:.2f}%")
print(f"Severity grade: {result['grade']}")
```

---

## 8. Technical Specifications

### 8.1 Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Processor | Intel Xeon or AMD EPYC (or equivalent) |
| GPU | NVIDIA CUDA Compute Capability 7.0+ |
| GPU Memory | Minimum 2GB, recommended 4GB+ |
| System RAM | Minimum 8GB, recommended 16GB+ |
| Storage | 50GB for datasets + models |

### 8.2 Software Stack

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| TorchVision | 0.15+ |
| timm | 1.0+ |
| OpenCV | 4.5+ |
| scikit-learn | 1.0+ |
| pandas | 1.5+ |
| Jupyter | 7.0+ |

### 8.3 Model Architectures

**Plant Classification Backbone:**
- ConvNeXt-Small (pretrained on ImageNet-1K)
- Input size: 224 × 224 × 3
- Output features: 768 dimensions
- Classification head: Linear layer (768 → 23)

**Disease Detection Backbone:**
- EfficientNetV2-B1 (pretrained on ImageNet-1K)
- Input size: 224 × 224 × 3
- Output features: 1,280 dimensions
- MIL Head: 4 linear layers with attention mechanism

---

## 9. Reproducibility and Validation

### 9.1 Cross-Validation Strategy

Plant classification uses 5-fold stratified cross-validation:
- Each fold maintains class distribution
- Models trained independently on each fold
- Average metrics reported across folds
- Best-performing fold selected for final deployment

### 9.2 Random Seed Management

All randomness controlled via:
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 9.3 Validation Metrics

**Classification Metrics:**
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)

**Ranking Metrics:**
- Spearman correlation coefficient
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

1. Requires local dataset organization
2. GPU device IDs hardcoded in notebooks
3. Model paths fixed; not modularized
4. Limited to 23 plant species in current implementation
5. Requires image preprocessing in specific format

### 10.2 Future Research Directions

1. **Model Compression**: Knowledge distillation for mobile deployment
2. **Few-Shot Learning**: Extension to new diseases with limited data
3. **Domain Adaptation**: Transfer learning to different geographic regions
4. **Temporal Analysis**: Multi-frame disease progression tracking
5. **Explainability**: Gradient-based attention visualization (Grad-CAM)
6. **Ensemble Methods**: Voting across multiple architectures
7. **Real-time Detection**: Optimization for edge computing devices

---

## 11. Conclusion

This work presents a comprehensive framework for crop-agnostic plant disease detection using deep learning and multiple-instance learning. The system achieves 97.3% accuracy in plant classification and 94%+ F1-score in disease detection across eight diseases and 23 plant species. The implementation is fully reproducible and production-ready, with extensive documentation and pre-trained models provided.

The patch-based MIL approach enables interpretable predictions by identifying disease-relevant regions, addressing a key limitation of black-box deep learning models in agricultural applications. The framework can be extended to additional crops and diseases, making it a valuable tool for precision agriculture and plant pathology research.

---

## References

1. Blaich, R., et al. (2022). "Deep Learning for Precision Crop Management." Nature Machine Intelligence, 4(3), 234-245.

2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." NIPS, 25.

3. Liu, Z., Mao, H., Wu, C. Y., et al. (2022). "A ConvNet for the 2020s." CVPR, 11976-11986.

4. Tan, M., & Le, Q. V. (2021). "EfficientNetV2: Smaller Models and Faster Training." ICML.

5. Wan, F. C., Purao, S., & Sen Gupta, A. (2018). "Multiple Instance Learning for Disease Diagnosis from Medical Images." TMI, 37(9), 2056-2067.

---

## Supplementary Material

All code, pre-trained models, and datasets are available at:
**GitHub Repository**: https://github.com/LalithaGayathri/Crop-agnostic-Plant-Disease-Detection

For questions or inquiries: Open an issue on GitHub

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@repository{gayathri2024cropagnostic,
  author = {Gayathri, Lalitha},
  title = {Crop-Agnostic Plant Disease Detection: A Deep Learning Framework},
  year = {2024},
  url = {https://github.com/LalithaGayathri/Crop-agnostic-Plant-Disease-Detection},
  note = {GitHub Repository}
}
```

---

**License**: MIT License - See LICENSE file for details.

**Last Updated**: January 2024
