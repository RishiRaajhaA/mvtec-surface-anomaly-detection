# 🔍 AI-Based Surface Anomaly Detection in Manufacturing

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Dataset](https://img.shields.io/badge/Dataset-MVTecAD-green)
![Task](https://img.shields.io/badge/Task-AnomalyDetection-orange)

Deep learning based **surface anomaly detection for industrial inspection** using **Autoencoders, Attention UNet and Variational Autoencoders** on the **MVTec AD dataset**.

This project explores **unsupervised defect detection** where models are trained only on **normal samples** and anomalies are detected via **reconstruction errors**.

---

# 📌 Overview

Automated visual inspection is a critical part of **quality control in modern manufacturing systems**.

Traditional rule-based computer vision approaches struggle with:

- diverse defect types
- complex textures
- varying lighting conditions

This project investigates **deep learning reconstruction-based anomaly detection** for detecting surface defects.

Three architectures are compared:

- **Simple Convolutional Autoencoder (SimpleAE)**
- **Attention UNet with CBAM and Attention Gates**
- **Convolutional Variational Autoencoder (ConvVAE)**

---

# 📂 Repository Structure

```
mvtec-surface-anomaly-detection
│
├── anomaly_detection.ipynb
└── README.md
```

The notebook contains the **complete implementation** including:

- preprocessing
- model architectures
- training
- anomaly scoring
- evaluation metrics
- visualization

---

# 📊 Dataset

This project uses the **MVTec Anomaly Detection Dataset**, a widely used benchmark for industrial defect detection.

📥 Download Dataset:

https://www.mvtec.com/company/research/datasets/mvtec-ad

After downloading, extract the **metal_nut** category.

Example dataset structure:

```
mvtec_ad
│
├── metal_nut
│   ├── train
│   │   └── good
│   ├── test
│   │   ├── good
│   │   ├── bent
│   │   ├── color
│   │   ├── flip
│   │   └── scratch
│   └── ground_truth
```

Training images contain **only normal samples**, while anomalies appear in the **test set**.

---

# ⚙️ Preprocessing Pipeline

Each image is converted into a **3-channel fused representation**:

1️⃣ Grayscale intensity  
2️⃣ Canny edge map  
3️⃣ Contour map  

This provides both **texture and structural information**.

Input size:

```
256 × 256
```

---

# 🧠 Model Architectures

## 1️⃣ Simple Convolutional Autoencoder

A lightweight encoder-decoder network with a bottleneck.

Advantages:

- parameter efficient
- strong baseline
- high anomaly recall

---

## 2️⃣ Attention UNet

Enhanced UNet architecture with:

- **CBAM (Convolutional Block Attention Module)**
- **Attention Gates**

Benefits:

- suppress irrelevant texture
- improve defect localization
- best image-level anomaly detection

---

## 3️⃣ Convolutional Variational Autoencoder

A probabilistic model that learns a **Gaussian latent distribution**.

Anomaly signals come from:

- reconstruction error
- latent KL divergence

---

# 📉 Loss Function

A **compound reconstruction loss** is used:

```
L = w1*MSE + w2*SSIM + w3*SobelEdge + w4*VGG_Perceptual
```

Components:

- **MSE Loss** → pixel accuracy  
- **SSIM Loss** → structural similarity  
- **Sobel Edge Loss** → boundary preservation  
- **VGG Perceptual Loss** → feature-level similarity  

---

# ⚡ Training Configuration

Optimizer:

```
AdamW
```

Learning rate:

```
1e-4
```

Scheduler:

```
Warmup Cosine LR
```

Batch size:

```
8
```

Additional techniques:

- gradient clipping
- early stopping
- data augmentation (flip, rotation)

---

# 📈 Anomaly Scoring

Multiple scoring methods were evaluated:

- MSE reconstruction error
- MSE + SSIM
- Top 2% highest error pixels
- Reconstruction + KL divergence (VAE)

The final anomaly score is obtained using **weighted combinations of these methods**.

---

# 📊 Results (metal_nut category)

| Model | Image AUROC | Pixel AUROC | Balanced Accuracy |
|------|------|------|------|
| SimpleAE | 0.6591 | 0.7598 | 0.6862 |
| Attention UNet | **0.8245** | 0.7748 | **0.7935** |
| ConvVAE | 0.7761 | **0.7956** | 0.7626 |

### Key Observations

- **Attention UNet achieves the best image-level anomaly detection**
- **ConvVAE provides the best pixel-level localization**
- **SimpleAE achieves highest sensitivity and F1 score**

---

# 🖼 Example Outputs

The notebook produces:

- reconstruction images
- error heatmaps
- ROC curves
- anomaly score distributions
- confusion matrices

These visualizations help interpret defect detection performance.

---

# 🚀 Future Work

Potential improvements:

- Test-Time Augmentation (TTA)
- Semi-supervised anomaly learning
- Multi-category MVTec training
- Edge-device deployment
- Model pruning and quantization

---

# 👨‍💻 Authors

**Rishi Raajha A**  
**Sakthi Pranav S**  
**Harshvardhan V**  
**Naveen Sankar R.S**

School of Artificial Intelligence  
Amrita Vishwa Vidyapeetham

---

# 📚 References

- Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- An & Cho, *Variational Autoencoder Based Anomaly Detection*, 2015
