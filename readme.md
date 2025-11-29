# Transformer-GAN for Time-Series Anomaly Detection (TranAD-GAN)

## 🎯 Overview

This project implements an unsupervised anomaly detection framework based on a **Transformer Encoder-Decoder** architecture acting as a Generator, trained adversarially with a Discriminator. The goal is to accurately detect deviations (anomalies) in multivariate time-series data by learning the "normal" reconstruction pattern.

The model is evaluated using **Synthetic Anomaly Injection** due to the lack of reliable ground truth labels, providing a robust measure of its true detection capability.

---

## 🛠️ 1. Architecture and Components

The core model combines a Transformer-based Generator with a simple MLP Discriminator.

### A. Generator (Transformer Encoder-Decoder)
The Generator is responsible for reconstructing the input time-series window. Its loss function encourages three properties: accurate reconstruction, meaningful latent representations, and fooling the discriminator.

* **Input:** Time series window $\mathbf{x} \in \mathbb{R}^{L \times D}$ (Length $\times$ Features).
* **Positional Encoding:** Adds sinusoidal signals to preserve sequential order.
* **Output:** Reconstructed window $\mathbf{\hat{x}}$.
* **Latent Vector ($\mathbf{z}$):** Extracted from the Encoder output for contrastive learning.

### B. Discriminator (MLP)
The Discriminator distinguishes between original **Real** windows and **Reconstructed** (Fake) windows.

### C. Training Loss (Composite)
The Generator is optimized using a weighted sum of three distinct losses:
1.  **Reconstruction Loss ($L_{rec}$):** $\text{MSE}(\mathbf{x}, \mathbf{\hat{x}})$. Minimizes reconstruction error.
2.  **Adversarial Loss ($L_{adv}$):** Binary Cross-Entropy (BCE). Minimizes $\text{log}(1 - D(\mathbf{\hat{x}}))$. Encourages realistic output.
3.  **Contrastive Loss ($L_{cont}$):** InfoNCE loss on latent vectors $\mathbf{z}_1$ and $\mathbf{z}_2$ (from two different views of the same window). Encourages similar inputs to have similar embeddings.

---

## 📈 2. Training Convergence & Reconstruction Quality

Training was performed for **50 epochs**. The loss curves demonstrate stability, and the reconstruction check confirms the Generator's ability to model normal data effectively.

### Training Loss Convergence

The Generator loss (Blue) steadily decreases, while the Discriminator loss (Orange) stabilizes, indicating a successful GAN training dynamic.


*(File: trainloss.png)*

### Reconstruction Check (Epoch 50)

The model accurately reconstructs a normal time-series segment (black line) with minimal deviation (cyan dashed line), indicating strong learning of the "normal manifold." The slight deviation near the end is a common characteristic of time-series reconstruction models.


*(File: qualitycheck.png)*

---

## 🔬 3. Evaluation & Anomaly Results

Evaluation is performed on synthetically injected anomalies. The anomaly score is the **Mean Squared Error (MSE)** between the input and its reconstruction.

### Score Distribution Analysis

The histogram below is the most critical diagnostic tool. The **Normal Data (Blue)** reconstruction errors are clearly separated from the **Anomaly Data (Red)** errors, which are pushed to the right side (higher MSE). This separation is the foundation of high detection accuracy.



[Image of Score Distribution]

*(File: rem.png)*

### Quantitative Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Best Standard F1** | **0.5343** | F1 Score at the optimal threshold. |
| **AUPRC (AUC)** | **0.5426** | Area Under the Precision-Recall Curve. |
| **Best PA F1 Score** | **0.5927** | F1 Score after applying Point Adjustment, which is more tolerant to detection latency. |

The scores suggest that while the model separates the distributions, there is significant overlap (as seen in the histogram), leading to a moderate AUPRC/F1 score (~0.54).

### Precision-Recall Curve

The PR curve shows the model's trade-off, achieving the highest precision around **Recall $\approx 0.35$**.



[Image of Precision-Recall Curve]

*(File: prcurve.png)*

### Anomaly Score Visualization

The compressed global view uses **Max Pooling (Bin Size 50)** to shrink the long time series while retaining anomaly peaks. The high red spikes correspond directly to the time windows where synthetic anomalies were injected, confirming the model's ability to localize deviations.


*(File: anamolyscore.png)*

---

## ⚙️ 4. Code Execution Details

### Preprocessing Summary
| Step | Value | Purpose |
| :--- | :--- | :--- |
| **Normalization** | Z-Score (Mean=0, Std=1) | Preserves outlier magnitude. |
| **Window Size** | 100 | Context length for the Transformer. |
| **Stride** | 10 | Overlapping windows for dense coverage. |
| **Augmentation** | Geometric Masking | Improves robustness and generalization. |

### Data Preparation
The code uses `get_clean_loaders()` to scan the `/kaggle/input` directory, identify files with the **most common feature dimension**, and split them into `train_loader` and `test_loader`.
```python
train_loader, test_loader, feat_dim, test_files = get_clean_loaders()
