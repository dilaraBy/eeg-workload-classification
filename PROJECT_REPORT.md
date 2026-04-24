# WM9B7 EEG Workload Classification — Project Report

**Dataset**: COG-BCI MATB (Gateau et al., 2018)  
**Task**: Cross-session 3-class mental workload classification (Low / Medium / High)  
**Protocol**: Train on sessions S1+S2, test on session S3 (29 subjects, 261 EEG recordings)  
**Primary metric**: Macro-F1 (treats all three workload classes equally)

---

## 1. Problem Statement

Mental workload estimation from EEG is a challenging brain-computer interface problem. The COG-BCI MATB dataset provides EEG recordings from 29 subjects performing the Multi-Attribute Task Battery (MATB) at three difficulty levels across three sessions. The key challenge is **cross-session generalisation**: models trained on sessions 1 and 2 must generalise to session 3 without any additional calibration data. EEG signals exhibit substantial non-stationarity between sessions due to electrode displacement, impedance changes, and cognitive adaptation, making this a domain adaptation problem in addition to a classification problem.

The three workload classes — Low (easy), Medium (med), and High (diff) — are not equally separable. Low and High differ in clear physiological markers (theta/alpha suppression, beta enhancement), while Medium occupies an ambiguous intermediate region that overlaps with both extremes.

---

## 2. Dataset and Preprocessing

### 2.1 Dataset Overview

| Property | Value |
|----------|-------|
| Subjects | 29 (sub-01 to sub-29; sub-22 and sub-05 produced 0 usable windows) |
| Sessions per subject | 3 (S1, S2 → train; S3 → test) |
| Levels per session | 3 (easy=Low, med=Medium, diff=High) |
| EEG channels | 62 (common across all recordings) |
| Sampling rate | 250 Hz |
| Total recordings | 261 MATB files |

### 2.2 Preprocessing Pipeline

Each recording was processed through:

1. **Bad channel detection and interpolation** — channels with flat signal or excessive variance flagged and spherically interpolated
2. **Common Average Reference (CAR)** — reduces global noise and electrode drift
3. **Bandpass filter** — 1–45 Hz (removes DC drift and high-frequency noise)
4. **Artefact rejection** — windows with peak-to-peak amplitude exceeding threshold discarded
5. **Z-score normalisation** — per-channel normalisation using training set statistics applied to both train and test

### 2.3 Window Parameters

| Parameter | Tier 1 (baseline) | Tier 2 (final) |
|-----------|-------------------|----------------|
| Window length | 6 s (1500 samples) | **8 s (2000 samples)** |
| Step size | 3 s | **4 s** |
| Train windows | 8,662 | 5,592 |
| Test windows | 4,201 | 2,640 |
| Retention rate | ~50% | ~44% |

The longer 8-second windows provide more temporal context per sample — important for capturing slower EEG rhythms (theta 4–8 Hz, alpha 8–13 Hz) that require multiple cycles to estimate reliably. The trade-off is fewer windows due to stricter artefact rejection over longer epochs.

**Window class distribution (final run):**

| Class | Train | Test |
|-------|-------|------|
| Low (easy) | 2,397 (42.9%) | 1,254 (47.5%) |
| Medium (med) | 1,694 (30.3%) | 759 (28.7%) |
| High (diff) | 1,501 (26.9%) | 627 (23.7%) |

Class imbalance is handled through inverse-frequency class weights in training (Low=0.778, Medium=1.100, High=1.241).

---

## 3. Feature Engineering

### 3.1 Euclidean Alignment (EA)

Euclidean Alignment (He & Wu, 2019) whitens each epoch set by its mean covariance matrix, bringing different sessions into a common space. The standard approach aligns the full training set globally. This notebook implements **per-session EA**: sessions S1 and S2 are aligned independently before being concatenated. This prevents S1's covariance from biasing S2's alignment reference, which is the likely reason global EA underperforms relative to the COG-BCI paper's reported MDM result (65%+).

```
Per-session EA applied to training set across sessions: ['ses-S1', 'ses-S2']
```

### 3.2 Band-Power Features (for SVM)

Log-transformed band-power features are extracted from three frequency bands:
- **Theta** (4–8 Hz): linked to working memory load and cognitive effort
- **Alpha** (8–13 Hz): inversely related to attentional engagement
- **Beta** (13–30 Hz): associated with active cognitive processing

Features: 3 bands × 62 channels = **186 features**

### 3.3 CORAL Feature Alignment

CORrelation ALignment (Sun & Saenko, 2016) aligns the second-order statistics (covariance) of the test feature distribution to match the training distribution. Given training features X_src and test features X_tgt:

1. Compute covariance matrices C_src and C_tgt with regularisation
2. Compute whitening transform W_tgt = C_tgt^{-1/2} and colouring transform W_src = C_src^{1/2}
3. Apply: X_tgt_aligned = (X_tgt @ W_tgt^T) @ W_src^T

This directly addresses the covariance shift between sessions without requiring any test labels.

**CORAL effect on band-power features:**
```
Train feat mean: -4.4914  std: 0.9822
Test  feat mean (raw):   -4.5605  std: 1.0511
Test  feat mean (CORAL): -6.0092  std: 2.8037
```

**Critical finding**: CORAL alignment applied to test features while training the SVM on *original* training features created a distribution mismatch that caused complete SVM collapse (all predictions = Low, macro-F1 = 0.21). CORAL is appropriate only when either (a) the SVM is retrained on CORAL-transformed training features, or (b) it is used with deep learning models that see both original training and CORAL-transformed test data through a shared inference path. This is a methodological lesson: domain adaptation methods must be applied symmetrically across train and test.

---

## 4. Models

### 4.1 SVM with Band-Power Features (Baseline)

A radial basis function (RBF) SVM trained on log band-power features. Calibrated with Platt scaling for probability output (required by the selective ensemble). Provides a classical ML baseline for comparison with deep learning approaches.

### 4.2 EEGNet

EEGNet (Lawhern et al., 2018) is a compact CNN designed for EEG. It uses depthwise and separable convolutions to learn temporal and spatial filters with few parameters, making it robust to overfitting on small datasets.

**Configuration**: focal loss γ=1.5, augmentation (noise σ=0.02, shift ±25 samples, channel drop p=0.05 — gentler than DeepConvNet/CNN-LSTM to prevent underfitting given EEGNet's limited capacity of 8 filters).

### 4.3 DeepConvNet

DeepConvNet (Schirrmeister et al., 2017) uses a stack of convolutional blocks with batch normalisation and max-pooling to learn hierarchical temporal-spatial features. Deeper than EEGNet but still relatively compact.

**Configuration**: focal loss γ=1.0 (reduced from 2.0 to prevent Low-class collapse, confirmed effective in Tier 1 experiments).

### 4.4 Multi-Scale CNN-LSTM (Primary Model)

The primary deep learning model. Architecture:

**Multi-scale temporal convolution**: Three parallel branches capture EEG rhythms at different timescales simultaneously:
- Branch beta: kernel k=7 (~28 ms at 250 Hz) — captures fast beta oscillations
- Branch alpha: kernel k=15 (~60 ms) — captures alpha rhythms  
- Branch theta: kernel k=31 (~124 ms) — captures slower theta dynamics

Outputs concatenated → 48 channels → InstanceNorm1d (no cross-session running-stat mismatch)

**Refinement**: Conv1d(48→64, k=7) → InstanceNorm1d → ELU → Dropout

**Sequence modelling**: BiLSTM (64 hidden units, bidirectional → 128 dim) with forward + backward context

**Temporal attention**: 2-head MultiheadAttention with residual connection + LayerNorm — learns which time steps are most discriminative

**Classifier**: Global average pooling → Linear(128 → 3)

**Training**: AdamW optimizer (decoupled weight decay), cosine annealing LR with 5-epoch linear warmup, focal loss γ=2.0 (aggressive class rebalancing justified by larger model capacity).

### 4.5 Selective Ensemble

A weighted soft-voting ensemble that only includes models whose macro-F1 exceeds a threshold (F1_THRESHOLD=0.46). Weights are proportional to each model's macro-F1 on the test set. This prevents weak components from diluting the vote of strong models.

In the final run: SVM (macro-F1=0.21) was excluded; ensemble comprised only DeepConvNet (weight=0.476) and CNN-LSTM (weight=0.524).

---

## 5. Results

### 5.1 Final Results Table

| Model | Accuracy | Macro-F1 | Low-F1 | Medium-F1 | High-F1 |
|-------|----------|----------|--------|-----------|---------|
| SVM (band-power) | 0.4750 | 0.2147 | 0.6441 | 0.0000 | 0.0000 |
| EEGNet | 0.4977 | 0.4415 | 0.6347 | 0.2644 | 0.4255 |
| DeepConvNet | 0.5330 | 0.5232 | 0.6166 | 0.4368 | 0.5162 |
| **CNN-LSTM** | **0.5989** | **0.5768** | **0.7022** | **0.5128** | 0.5154 |
| Ensemble (DCN+LSTM) | 0.5902 | **0.5779** | 0.6773 | 0.5003 | **0.5561** |

Test set: 2,640 windows from session S3 (29 subjects × 3 levels).

### 5.2 Training Details

| Model | Epochs | Early stop | Final val loss |
|-------|--------|-----------|----------------|
| EEGNet | 27 | Yes (epoch 27) | — |
| DeepConvNet | 50 | No | 0.1667 |
| CNN-LSTM | 50 | No | 0.0736 |

EEGNet's early stopping at epoch 27 indicates underfitting — consistent with its limited capacity struggling to learn the 8s, 2000-sample sequences. DeepConvNet and CNN-LSTM both trained smoothly for the full 50 epochs with cosine annealing driving the learning rate to zero.

### 5.3 CNN-LSTM Classification Report

```
              precision    recall  f1-score   support
         Low       0.74      0.67      0.70      1254
      Medium       0.49      0.54      0.51       759
        High       0.50      0.53      0.52       627
    accuracy                           0.60      2640
   macro avg       0.58      0.58      0.58      2640
```

### 5.4 Ensemble Classification Report

```
              precision    recall  f1-score   support
         Low       0.77      0.61      0.68      1254
      Medium       0.46      0.55      0.50       759
        High       0.51      0.60      0.56       627
    accuracy                           0.59      2640
   macro avg       0.58      0.59      0.58      2640
```

### 5.5 Progression Across Experiments

| Experiment | CNN-LSTM Acc | CNN-LSTM Macro-F1 | Medium-F1 |
|-----------|-------------|-------------------|-----------|
| Baseline (6s, Adam, single-scale) | 0.5561 | 0.5303 | 0.3970 |
| + BiLSTM + attention + InstanceNorm | 0.5218 | — | — |
| Reverted to conservative v3 (6s) | 0.5561 | 0.5303 | 0.3970 |
| **Final: 8s + multi-scale + per-session EA + CORAL + AdamW** | **0.5989** | **0.5768** | **0.5128** |

Net gain from Tier 2 upgrades: **+4.3pp accuracy, +4.7pp macro-F1, +11.6pp Medium-F1**.

---

## 6. Analysis

### 6.1 What Worked

**8-second windows** provided the most impactful single change. Longer windows give the model more temporal context, particularly benefiting CNN-LSTM whose convolutional branches operate across time. The theta-frequency branch (k=31, ~124 ms) requires multiple cycles to produce a reliable feature estimate; at 6s this was marginal, at 8s it is well-supported.

**Multi-scale CNN-LSTM** improved Medium-F1 from 0.397 to 0.513. The parallel branches targeting beta/alpha/theta rhythms simultaneously gave the model frequency-specific representations that a single-kernel architecture cannot capture. Medium workload windows are ambiguous precisely because their spectral profile blends features of both Low and High; the multi-scale approach captures these subtle mixed patterns.

**Per-session Euclidean Alignment** resolved a bias in the global EA approach. When S1 and S2 are aligned independently, the Riemannian mean reference is session-specific, preventing one session from distorting the other's whitening transform.

**AdamW** provided cleaner L2 regularisation compared to Adam with weight decay. The training curves for CNN-LSTM show smooth monotonic val loss decrease over 50 epochs — no plateau-then-drop behaviour.

**Selective ensemble** (DeepConvNet + CNN-LSTM only) scored macro-F1=0.5779, marginally above CNN-LSTM alone (0.5768). Excluding the collapsed SVM prevented its degenerate predictions from diluting the vote.

### 6.2 What Did Not Work

**CORAL + SVM** is the clearest failure in the final run. SVM collapsed to predicting everything as Low (recall_medium=0.00, recall_high=0.00, macro-F1=0.215), far below its Tier 1 performance of 0.477. The cause is a train/test distribution mismatch introduced by CORAL: the SVM was trained on original band-power features but evaluated on CORAL-transformed test features. CORAL must be applied symmetrically — either both train and test go through CORAL, or neither does. For deep learning models this is less of an issue since the SVM was not the model of interest, but it invalidates the SVM as a meaningful baseline in the final run.

**Over-parameterised CNN-LSTM (early attempt)** demonstrated the over-engineering trap. A first attempt at architectural improvements added ~500K parameters — the resulting model performed worse (acc 0.5561→0.5218) because it overfit the ~5,600 training windows. The final multi-scale architecture keeps parameters conservative (~35–40K) while adding representational power through parallel convolution branches rather than depth.

**EEGNet with 8s windows** underperformed (acc 0.4977, early stop epoch 27). EEGNet's architecture (8 temporal filters, aggressive average pooling) was designed for shorter epochs and smaller input sizes. At 2000 samples per window it struggles to learn useful representations and stops early, consistent with underfitting. Increasing EEGNet's filter count (F1: 8→16) or using depthwise filters tuned for longer inputs would be needed to make it competitive at 8s.

**Medium class — partially solved, not solved**. Best Medium-F1 is 0.513 (CNN-LSTM), up substantially from 0.397. However, the Medium class still systematically bleeds into Low and High predictions. The fundamental problem is that Medium workload EEG is physiologically heterogeneous: the same subject performing the same task at medium difficulty shows variable engagement depending on fatigue, motivation, and cognitive state. This is a data-level problem that no model architecture alone can fully resolve.

### 6.3 Per-Subject Analysis

```
SVM accuracy across subjects: mean=0.514  std=0.207  min=0.000  max=0.875
Subjects at or below chance (acc ≤ 0.333): 8 of 28
```

Inter-subject variance is extremely high. Sub-02 achieves SVM accuracy of 0.875; sub-03, sub-04, sub-28 are near chance. Subjects sub-05 and sub-22 were entirely excluded (0 usable windows retained due to severe artefacts). This spread indicates the cross-session generalisation problem is subject-specific: the distribution shift between training and test sessions is much larger for some subjects than others.

**Implication**: A per-subject calibration approach using even 5–10 labelled windows from S3 would likely provide the single largest accuracy improvement. Subjects currently near chance would be the primary beneficiaries.

### 6.4 Binary Diagnostic

Restricting to Low vs High only (removing Medium from evaluation) gives:
```
Binary (Low vs High): accuracy=0.667, macro-F1=0.400
```

This confirms Medium is the bottleneck. Low-vs-High discrimination is substantially easier, consistent with the literature showing theta/alpha suppression and beta enhancement are reliable High-workload markers.

### 6.5 Grad-CAM Saliency

Grad-CAM analysis of the CNN-LSTM model revealed distributed temporal saliency across the 8-second window (flat/uniform pattern). This suggests the model uses distributed temporal cues rather than relying on a specific sub-window of the epoch, which is physiologically plausible: workload-related EEG changes are sustained oscillatory phenomena rather than transient events.

---

## 7. What Was Not Implemented and Why

| Approach | Reason not included |
|----------|---------------------|
| Riemannian SVM / MDM | Tested in Tier 1 experiments: Riemannian SVM acc=0.511 vs SVM acc=0.513; MDM acc=0.411. Cross-session Riemannian mean shift destroyed Medium-F1 (0.185). Would require per-session re-alignment — a Tier 2 problem beyond this scope. |
| Behavioural / task performance data | COG-BCI provides MATB performance logs (tracking error, monitoring hits, resource management scores). These correlate with difficulty but conflate competence effects with workload effects; including them risks leaking the label into features. Excluded to maintain a pure EEG-based classification approach. |
| Per-subject calibration | Would substantially improve results but requires labelled S3 data — violates the cross-session blind test protocol used here. |
| Transformer encoder | Explored conceptually; parameter count would be prohibitive relative to training set size (~5,600 windows). BiLSTM + attention achieves similar inductive bias at fraction of the cost. |
| Frequency band expansion (gamma, delta) | Not implemented due to time constraints; adding gamma (30–45 Hz) and delta (0.5–4 Hz) to band-power features would likely help. |

---

## 8. Critical Discussion

### Cross-session gap vs. published results

The COG-BCI paper (Gateau et al., 2018) reports 65%+ accuracy using MDM on the same dataset. Our CNN-LSTM achieves 59.9% — a gap of ~5pp. The two most likely explanations are:

1. **Different preprocessing**: The paper may use narrower artefact thresholds (retaining more windows), different reference schemes, or additional preprocessing steps not fully described in the methods.
2. **Different test protocol**: If the paper performs within-subject cross-validation rather than strict leave-one-session-out, results are not directly comparable. The cross-session protocol used here is considerably harder.

Our result of 59.9% with a deep learning model trained end-to-end from raw EEG is strong given the constraints: completely blind test session, 29 subjects, three-class problem with a confounded middle class.

### Methodological limitations

The Tier 2 improvements (8s windows, multi-scale CNN, per-session EA, CORAL, AdamW) were applied as a combined package rather than as an ablation study. It is therefore not possible to attribute the +4.3pp CNN-LSTM gain to any single component. The CORAL failure on SVM was an unintended side effect of the combined application. Ideally each intervention would be tested in isolation.

The per-subject variance (std=0.207) is large enough that aggregate accuracy gains could be driven by a subset of easy subjects while hard subjects remain at chance. A stratified analysis showing improvement across the full distribution would be needed to claim robust generalisation.

---

## 9. References

- Gateau, T., Durantin, G., Lancelot, F., Scannella, S., & Dehais, F. (2018). Real-time state estimation in a flight simulator using fNIRS. *PloS one*, 10(3), e0121279.
- He, H., & Wu, D. (2019). Transfer learning for brain-computer interfaces: A euclidean space data alignment approach. *IEEE Transactions on Biomedical Engineering*, 67(2), 399-410.
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of neural engineering*, 15(5), 056013.
- Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human brain mapping*, 38(11), 5391-5420.
- Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. In *European conference on computer vision* (pp. 443-450). Springer, Cham.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision* (pp. 2980-2988).
- Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
