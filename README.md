# Deepfake Detection using Spatial–Frequency Fusion with Region-Level Localization

##  Overview
This project explores a **hybrid deepfake detection framework** that combines spatial (RGB), frequency (FFT), and transformer-based reasoning to improve robustness and interpretability.

Unlike traditional detectors that rely purely on visual artifacts, this system integrates **frequency-domain analysis and region-level reasoning**, enabling both **accurate classification and explainable localization of manipulated regions**.

---

##  Key Idea

Modern deepfake detectors suffer from:
- Poor **cross-dataset generalization**
- Lack of **interpretability**
- Weak integration of **frequency-domain signals**

This project addresses these gaps by combining:
- CNNs for spatial features
- FFT for frequency artifacts
- Transformers for global reasoning
- Superpixel-based segmentation for region-level explainability

---

##  Methodology

###  Stage 1 — Frequency-Aware Baseline
- Applied **Fast Fourier Transform (FFT)** to extract spectral features
- Built a **dual-stream CNN** processing:
  - RGB images
  - Log-scaled FFT magnitude
- Demonstrated that frequency signals capture **compression-resistant artifacts** :contentReference[oaicite:0]{index=0}

---

###  Stage 2 — Cross-Modal Fusion
- Combined spatial and frequency features using:
  - **Cross-attention mechanisms**
  - Transformer-based token fusion
- Inspired by models like M2TR but implemented with:
  - ResNet backbone
  - Custom frequency filtering

---

###  Stage 3 — Region-Level Reasoning (THIS SEMESTER )

This semester focused on **segmentation + localization**.

#### Problem:
Standard models only predict:
> “this image is fake”

But do NOT answer:
> “where is it fake?”

---

###  Solution: Region-Based Localization

Instead of segmenting FFT (incorrect approach), we:

#### Step 1 — Spatial Segmentation
- Generated **SLIC superpixels on RGB images**
- Created meaningful spatial regions (face-aware)

#### Step 2 — Dual Evidence Extraction
For each region:
- **RGB evidence** → Grad-CAM from CNN backbone
- **FFT evidence** → frequency-aware activations / fusion signals

#### Step 3 — Region Scoring
Each region receives:
- RGB fake score
- FFT fake score
- Combined anomaly score

#### Step 4 — Pseudo Label Generation
- High-score regions → fake
- Low-score regions → real
- Used as **weak supervision signals**

#### Step 5 — Region Prediction Head
Extended model to:
- Predict **image-level label**
- Predict **region-level fake probabilities**

---

##  Training Strategy

- Pretraining on large-scale dataset (140k images)
- Fine-tuning on FaceForensics++
- Careful FFT placement (raw vs normalized input)


---

##  Evaluation

We evaluate:
- Image-level metrics (Accuracy, AUROC, F1)
- Cross-dataset generalization
- Region-level interpretability via:
  - Fake region overlays
  - Coverage % of manipulated areas

---
 
##  Key Contributions

-  Hybrid **RGB + FFT + Transformer architecture**
-  **Cross-modal attention fusion**
-  **Region-level deepfake localization**
-  Weakly-supervised segmentation pipeline
-  Improved interpretability beyond Grad-CAM

---

##  Important Insights

- FFT should be applied **before normalization** for meaningful frequency signals :contentReference[oaicite:1]{index=1}  
- Frequency signals are **global**, not region-based → segmentation must stay in RGB space :contentReference[oaicite:2]{index=2}  
- Grad-CAM alone is insufficient → model must be **trained to localize**  

---

##  Future Work

- Face-part tokenization instead of SLIC
- Video-based detection (temporal modeling)
- Stronger supervision for segmentation
- Cross-generator generalization

---

##  Tech Stack

- Python, PyTorch
- OpenCV, NumPy
- ResNet50, Vision Transformers
- FFT-based feature extraction
- Grad-CAM & attention visualization

---

##  Final Takeaway

This project moves beyond “black-box classification” toward:
> **Explainable, frequency-aware deepfake detection systems that can both detect and localize manipulation.**
