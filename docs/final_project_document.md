# WM9B7 EEG Workload Classification - Complete Step-by-Step Project Document

## 1. Project Definition

### 1.1 Problem
Classify mental workload from MATB EEG into three classes:
- `Low` (MATBeasy)
- `Medium` (MATBmed)
- `High` (MATBdiff)

### 1.2 Research Question
Can a deep learning model classify 3-level workload from MATB EEG better than a traditional ML baseline under cross-session evaluation?

### 1.3 Why this project is valid for WM9B7
- Uses open-source EEG dataset and reproducible notebook workflow.
- Directly supports LO1/LO2/LO3 via ML-vs-DL comparison, critical evaluation, and future-trend discussion.
- Not in transportation/mobility scope.

## 2. Dataset and Evidence Baseline

### 2.1 Current verified dataset status
From EDA:
- 261 MATB `.set` files.
- 29 subjects, 3 sessions, 3 classes.
- Balanced file counts by class/session.
- Continuous recordings (~299s each), 500 Hz.
- Channel mismatch exists (63 vs 64 channels), only `Cz` differs.

### 2.2 Why this matters
These findings determine preprocessing design:
- We must harmonize channels before modeling.
- We must convert continuous recordings into fixed windows.
- We must use cross-session-safe normalization and split logic.

## 3. End-to-End Project Structure

## Phase A - Scope Lock and Submission Architecture
### Step A1 - Freeze objective and split protocol
What to do:
- Lock project scope to MATB-only workload classification.
- Lock split protocol: train on `S1+S2`, test on `S3`.
- Select one primary submission notebook entrypoint.

Why:
- Prevents scope drift and protects evaluation validity.

Outputs:
- Final objective statement.
- Fixed split statement in notebook and report.

## Phase B - Reproducibility Layer
### Step B1 - Data source strategy
What to do:
- Develop locally first for speed.
- Implement dual-source loader:
  - local mode (development)
  - remote mode (Google Drive/fallback mirror) for final Run All validation.

Why:
- Fast iteration now and compliance with marker reproducibility expectations.

Outputs:
- Config flag and loader cell.
- Download/unzip checks and clear error handling.

## Phase C - EDA Finalization
### Step C1 - Data understanding notebook
What to do:
- Keep EDA outputs: class/session counts, channel consistency, duration/sfreq checks, signal preview, PSD, annotation examples.

Why:
- Converts assumptions into evidence and justifies preprocessing choices.

Outputs:
- EDA notebook figures/tables.
- EDA findings markdown summary.

## Phase D - Labeling and Split Integrity
### Step D1 - Label map
What to do:
- Map:
  - `MATBeasy -> 0`
  - `MATBmed -> 1`
  - `MATBdiff -> 2`

Why:
- Deterministic labels for reproducible training/evaluation.

Outputs:
- Label mapping cell with explicit printout.

### Step D2 - Cross-session split assertions
What to do:
- Build split function with assertions:
  - train uses only `S1/S2`
  - test uses only `S3`
  - no leakage.

Why:
- Prevents inflated performance and invalid conclusions.

Outputs:
- Split integrity logs and class distribution by split.

## Phase E - Chosen Preprocessing Pipeline (Core)

### Step E1 - Channel harmonization
What to do:
- Use common channel intersection across files.
- Drop `Cz` globally to keep a stable 63-channel representation.

Why:
- Avoids shape mismatch and model input inconsistency.

Outputs:
- Harmonized channel list and channel count confirmation.

### Step E2 - Signal cleaning and referencing
What to do:
- Set montage (10-20 compatible).
- Notch filter at 50 Hz.
- Band-pass filter 1-40 Hz (FIR, zero-phase).
- Detect bad channels (variance/robust z-score) and interpolate.
- Re-reference to common average.

Why:
- Removes line noise and irrelevant frequencies.
- Stabilizes channel quality and spatial reference.

Outputs:
- Preprocessing log per file (bad channel counts, interpolation applied).

### Step E3 - Resampling and epoching
What to do:
- Resample from 500 Hz to 250 Hz.
- Create 2-second windows with 50% overlap within each MATB block.
- Reject epochs with peak-to-peak amplitude > 150 microvolts.

Why:
- 250 Hz reduces compute while preserving workload-relevant EEG dynamics.
- Overlap increases sample efficiency.
- Threshold rejection controls high-amplitude contamination.

Outputs:
- `X_train`, `y_train`, `X_test`, `y_test` with shape and retention summary.

### Step E4 - Standardization
What to do:
- Fit z-score normalization on training data only (per channel).
- Apply same stats to test data.

Why:
- Prevents information leakage while controlling scale differences.

Outputs:
- Stored normalization parameters and normalized arrays.

### Step E5 - Optional ICA path (ablation)
What to do:
- Keep ICA optional as an improvement experiment, not default core pipeline.

Why:
- ICA may improve artifact handling but increases runtime/complexity risk for final Run All.

Outputs:
- Optional metrics comparison (core vs core+ICA).

## Phase F - Cross-Session Robustness Enhancement
### Step F1 - Euclidean Alignment (EA)
What to do:
- Apply EA as an additional preprocessing branch.
- Compare metrics with and without EA.

Why:
- Directly targets session covariance shift observed in cross-session failure modes.

Outputs:
- Ablation table: baseline preprocessing vs baseline+EA.

## Phase G - Model Baseline 1 (Traditional ML)
### Step G1 - SVM pipeline
What to do:
- Extract band-power features (theta/alpha/beta).
- Train `StandardScaler + RBF SVM`.

Why:
- Provides interpretable, classical baseline for LO1.

Outputs:
- Accuracy, macro-F1, per-class recall, confusion matrix, classification report.

## Phase H - Model Baseline 2 (Deep Learning)
### Step H1 - EEGNet training
What to do:
- Train EEGNet with:
  - class weighting,
  - early stopping,
  - LR scheduler,
  - fixed random seeds.

Why:
- Compact EEG-specific architecture suitable for coursework compute and direct ML-vs-DL comparison.

Outputs:
- Learning curves, best model metrics, confusion matrix, per-class recalls.

## Phase I - Optional Model Extension
### Step I1 - CNN-LSTM (recommended optional)
What to do:
- Train CNN-LSTM once core SVM+EEGNet pipeline is stable.

Why:
- Adds temporal modeling depth for stronger technical analysis.

Outputs:
- Extra row in model comparison table.

### Step I2 - LaBraM (stretch)
What to do:
- Attempt only if all core deliverables are complete and reproducible.

Why:
- High complexity; should not risk baseline submission quality.

Outputs:
- Optional advanced comparison and future-trend evidence.

## Phase J - Evaluation and Critical Analysis
### Step J1 - Unified evaluation block
What to do:
- Report for each model:
  - Accuracy
  - Macro F1
  - Per-class precision/recall
  - Confusion matrix

Why:
- Needed for fair model comparison and report evidence.

Outputs:
- Final model comparison table.

### Step J2 - Error analysis
What to do:
- Analyze class confusion, especially medium class behavior.
- Relate likely causes to EEG non-stationarity and class overlap.

Why:
- Required for critical reflection quality (LO2).

Outputs:
- 3-5 evidence-backed interpretation points.

## Phase K - Reflection Report Build (LO-aligned)

### Step K1 - LO1 section (ML vs DL)
What to do:
- Compare representational assumptions, feature engineering burden, generalization behavior, and trade-offs.

Why:
- Directly addresses LO1 criteria.

Outputs:
- Structured comparison with references and project metrics.

### Step K2 - LO2 section (critical evaluation + implications)
What to do:
- Justify architecture and preprocessing decisions.
- Discuss limitations (chance-level risk, medium-class weakness, cross-session drift).
- Discuss societal/organizational implications, ethics, and monitoring caveats.

Why:
- Meets LO2 depth and criticality expectations.

Outputs:
- Decision-justification and limitations table.

### Step K3 - LO3 section (future trends)
What to do:
- Link future opportunities to this project:
  - foundation models,
  - domain adaptation,
  - multimodal EEG+ECG,
  - explainability.

Why:
- Satisfies LO3 with relevant forward-looking argument.

Outputs:
- Evidence-backed future-work section.

## Phase L - Submission Hardening
### Step L1 - Reproducibility final run
What to do:
- Run notebook from clean kernel in remote data mode.
- Verify all cells execute without manual intervention.

Why:
- Prevents submission-time reproducibility failure.

Outputs:
- Final run log and runtime note.

### Step L2 - Compliance checks
What to do:
- Confirm dataset citation/license in notebook/report.
- Confirm AI collaboration appendix.
- Confirm file-count and accepted format limits.

Why:
- Avoids avoidable grading penalties.

Outputs:
- Submission-ready checklist completion.

## 4. Final Model Strategy (Decision)

Mandatory core:
1. `SVM + band-power`
2. `EEGNet`

Recommended optional:
3. `CNN-LSTM`

Stretch only:
4. `LaBraM`

Decision rule:
- If runtime/reproducibility risk appears, keep only SVM+EEGNet and strengthen analysis quality.

## 5. Verification Checklist (Must Pass)

1. Data integrity:
- full MATB coverage and balanced class/session counts confirmed.

2. Preprocessing integrity:
- channel harmonization applied and logged.
- epoch retention rate reported.

3. Split integrity:
- strict cross-session split assertions passed.

4. Model evidence:
- metrics and confusion matrices generated for SVM and EEGNet at minimum.

5. Reproducibility:
- clean-kernel Run All in remote mode successful.

6. Report quality:
- each key claim tied to a metric, figure, or citation.

## 6. Suggested Deliverable File Set (<= 5 files)

1. Main notebook (`.ipynb`) - full reproducible pipeline.
2. Critical reflection report (`.pdf` or `.docx`).
3. Optional supporting script (`.py`) if essential.
4. Optional metrics table export (`.csv` or `.txt`) if referenced.
5. Optional AI-usage appendix (`.txt`/included in report) per policy.

## 7. Immediate Next Actions (Execution Order)

1. Add finalized preprocessing markdown + code block into the main notebook.
2. Implement split + preprocessing core pipeline and produce train/test arrays.
3. Train SVM baseline and EEGNet, generate comparison outputs.
4. Draft report sections in parallel using generated evidence.
5. Perform remote-source reproducibility run and finalize submission package.
