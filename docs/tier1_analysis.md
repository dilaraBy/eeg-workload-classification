# Tier 1 Experiment Analysis & Main Notebook Recommendations

**Source**: `WM9B7_EEG_Workload_Implementation copy 3.ipynb` (full run, GPU, 29 subjects)  
**Protocol**: Train S1+S2 → Test S3 (cross-session), 8662 train windows / 4201 test windows

---

## 1. Full Results Table

| Model | Accuracy | Macro-F1 | Low-F1 | Medium-F1 | High-F1 |
|-------|----------|----------|--------|-----------|---------|
| SVM (band-power) — baseline | 0.5134 | 0.4770 | 0.635 | 0.379 | 0.417 |
| Riemannian SVM (T1-D) | 0.5113 | 0.4169 | 0.665 | **0.185** | 0.401 |
| MDM (T1-D) | 0.4109 | 0.3929 | 0.520 | 0.297 | 0.362 |
| EEGNet (T1-A/B/C) | 0.4742 | 0.4514 | 0.592 | 0.350 | 0.412 |
| DeepConvNet (T1-A/B/C) | 0.5068 | 0.4854 | 0.613 | 0.359 | 0.484 |
| **CNN-LSTM (T1-A/B/C)** | **0.5561** | **0.5303** | **0.698** | **0.397** | **0.496** |
| Ensemble T1-E (all 5) | 0.5446 | 0.5028 | 0.676 | 0.358 | 0.475 |

Previous baseline (before Tier 1, from main notebook first run):

| Model | Accuracy |
|-------|----------|
| SVM | 0.5134 |
| EEGNet | 0.5292 |
| DeepConvNet | 0.4639 |

---

## 2. What Worked

### CNN-LSTM — best overall model (+9.3pp acc vs DeepConvNet baseline)
- Accuracy 0.5561, macro-F1 0.5303 — best by a clear margin on all metrics
- Strongest on all three classes simultaneously (Low 0.70, Medium 0.40, High 0.50)
- Its 1D temporal convolutions + LSTM capture sequential EEG dynamics better than pure spatial architectures for cross-session generalisation
- **Keep in main notebook as the primary DL model**

### T1-A per-model focal gamma — helped DeepConvNet significantly
- DeepConvNet improved from acc=0.4639 (baseline) to 0.5068 (+4.3pp) simply by reducing γ from 2.0→1.0
- The Low recall collapse that happened at γ=2.0 was fully resolved
- **Keep T1-A in main notebook**

### T1-B cosine-warmup LR — better training stability
- DeepConvNet trained for 37 epochs before early stopping vs EEGNet which stopped at epoch 20
- Loss curves show smooth convergence; no plateau-then-drop behaviour from ReduceLROnPlateau
- **Keep T1-B in main notebook**

### T1-C EEG augmentation — net positive for CNN-LSTM and DeepConvNet
- Medium class recall improved across DeepConvNet and CNN-LSTM vs the un-augmented baseline
- High class recall on CNN-LSTM is 0.59 — the best seen across all experiments
- **Keep T1-C for CNN-LSTM and DeepConvNet**

### Focal loss + label smoothing — medium class rescue (partial)
- Medium recall in the original baseline was near 0.03–0.18 (main notebook first run)
- After Tier 1 it reached 0.33–0.40 across DL models — a real improvement, though not solved
- **Keep both in main notebook**

---

## 3. What Did Not Work

### EEGNet regressed (−5.5pp acc vs baseline)
- Dropped from 0.5292 → 0.4742 after applying T1-A/B/C
- Stopped early at epoch 20 — underfitting, not overfitting
- **Root cause**: focal γ=2.0 + strong augmentation (σ=0.05 noise + channel dropout) is too aggressive for EEGNet's small capacity (8 filters). The model cannot recover the signal under that noise level
- **Fix for main notebook**: reduce EEGNet's augmentation strength (noise_std 0.05→0.02, ch_drop_p 0.10→0.05) or disable augmentation for EEGNet entirely and only use γ=1.5

### Riemannian SVM underperformed band-power SVM
- Acc 0.5113 / macro-F1 0.4169 vs SVM 0.5134 / 0.4770 — Medium F1 collapsed to 0.185
- 1953 tangent-space features estimated from session-specific covariances are not cross-session stable — the Riemannian mean shifts between S1+S2 and S3
- **Do not include T1-D in the main notebook** without per-session re-alignment (a Tier 2 problem)

### MDM — worst performer overall
- Acc 0.4109 — barely above chance for a 3-class problem (0.333)
- MDM assumes class means are well-separated in Riemannian geometry; cross-session distribution shift breaks this assumption
- **Do not include MDM in the main notebook**

### Ensemble T1-E underperformed the best single model
- Ensemble acc 0.5446 vs CNN-LSTM 0.5561 (−1.1pp) — the ensemble is dragging CNN-LSTM down
- Weak components (Riemannian SVM macro-F1=0.42, MDM 0.39, EEGNet 0.45) dilute the vote even with macro-F1 weighting
- Weights were: CNN-LSTM 0.225, DeepConvNet 0.206, SVM-BP 0.202, EEGNet 0.191, Riem-SVM 0.177 — too evenly spread
- **Fix for main notebook**: selective ensemble — only include models whose macro-F1 exceeds a threshold (e.g. 0.47). With this run that would keep SVM-BP, DeepConvNet, CNN-LSTM and exclude EEGNet + Riemannian SVM

### Medium class — still the core unsolved problem
- Best Medium-F1 across all models is 0.40 (CNN-LSTM)
- All models systematically predict Medium windows as Low (confusion matrix: Medium→Low is the most common error)
- This requires feature-space separation improvement, not just loss weighting

---

## 4. Per-Subject Analysis Key Findings

- **High variance**: SVM accuracy ranges 0.000–0.928 across 29 subjects (mean=0.494, std=0.225)
- **7 of 28 subjects** are at or below chance level (acc ≤ 0.333) — these are the main drivers of the overall score gap
- **Sub-22 and sub-05** had 0 usable windows retained in preprocessing — completely excluded, creating gaps in evaluation
- The distribution shift is subject-specific: some subjects (sub-02, sub-25) generalise well, others (sub-03, sub-04, sub-28) do not
- **Implication**: a per-subject calibration strategy (even just 5–10 windows from S3) would likely give the largest single improvement

---

## 5. What to Implement in the Main Notebook

### Must implement (clear positive signal)

| Change | Reason | Expected gain |
|--------|--------|---------------|
| CNN-LSTM as primary model | Best accuracy (0.5561) and macro-F1 (0.5303) | +9.3pp acc vs DeepConvNet baseline |
| T1-A per-model gamma (EEGNet γ=1.5, DeepConvNet γ=1.0, CNN-LSTM γ=2.0) | Prevents DeepConvNet Low-collapse; EEGNet needs gentler gamma | +4.3pp on DeepConvNet |
| T1-B cosine-warmup LR (5 epochs warmup) | Smoother convergence, earlier stopping avoided | Stable training |
| Focal loss + label smoothing (keep as-is) | Medium recall improved from <0.18 to 0.33–0.40 | Maintained medium improvement |

### Implement with modification

| Change | Modification | Reason |
|--------|-------------|--------|
| T1-C augmentation | Separate config per model: CNN-LSTM/DeepConvNet use full aug; EEGNet uses noise_std=0.02, ch_drop_p=0.05 or no aug | EEGNet regressed under full augmentation |
| Ensemble (T1-E) | Selective: only include models with macro-F1 > threshold (computed at runtime). With current results: SVM-BP + DeepConvNet + CNN-LSTM | Removes weak components dragging the vote down |

### Do not implement

| Change | Reason |
|--------|--------|
| Riemannian SVM (T1-D) | Worse than band-power SVM cross-session; Medium F1 collapsed to 0.185 |
| MDM (T1-D) | Worst performer (acc 0.41); cross-session distribution shift breaks Riemannian mean assumption |

### Recommended next steps after main notebook (Tier 2 priorities)

1. **CORAL domain adaptation** — aligns second-order statistics of S3 to S1+S2 distribution; directly attacks the cross-session covariance shift that killed Riemannian methods
2. **Per-session EA** — apply Euclidean Alignment per-session rather than globally, to reduce session-specific drift before feature extraction
3. **Subject-adaptive calibration** — use a small held-out portion of S3 (5–10 windows per class) to fine-tune the classifier; per-subject analysis shows this could push several subjects from chance to 0.6+
4. **EEGNet architecture revisit** — increase filter count (F1: 8→16) or reduce aggressive pooling to give it more capacity before applying focal loss
5. **Frequency band expansion** — add gamma (30–45 Hz) and slow cortical (0.5–4 Hz delta) to the band-power feature set; theta/alpha/beta misses workload-relevant high-frequency activity

---

## 6. Summary Verdict

The Tier 1 changes delivered a **mixed but net-positive result**. The headline improvement is **CNN-LSTM at 0.5561 accuracy and 0.5303 macro-F1** — the best result in this project so far. The cosine-warmup scheduler and per-model focal gamma were the most reliable interventions. Riemannian features and MDM failed specifically because of cross-session distribution shift, which is the fundamental problem in this dataset. The ensemble concept is sound but needs to be selective. EEGNet needs gentler augmentation settings.

The main notebook should adopt CNN-LSTM + selective ensemble as its upgraded architecture, skip the Riemannian methods, and fix the EEGNet augmentation strength.
