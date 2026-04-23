# Improvement Experiment Review and Main Notebook Decisions

Date: 2026-04-23
Source notebook: notebooks/WM9B7_EEG_Workload_Implementation copy.ipynb

## 1. Executive Summary
The experiments show that the biggest limitation is class overlap around the Medium workload label, not only model architecture.

What worked best overall:
- Tuned DeepConvNet is the strongest 3-class model in this round.
- Longer windows improved results in the tested setup (6 s > 4 s > 2 s on validation macro-F1, and 6 s had best test metrics among the sweep models).
- Binary Low vs High became very strong, confirming overlap difficulty is concentrated in the Medium class.

What did not work:
- CNN-LSTM variants underperformed compared with SVM and tuned DeepConvNet.
- Simply adding sequence complexity did not resolve Medium-class confusion.

## 2. Key Results
### 2.1 Main 3-class models (copy notebook)
- SVM + Band-Power: accuracy = 0.5451, macro-F1 = 0.5282
- DeepConvNet: accuracy = 0.4633, macro-F1 = 0.4387
- CNN-LSTM: accuracy = 0.3802, macro-F1 = 0.3658

### 2.2 Binary diagnostic (Low vs High)
- Binary baseline: accuracy = 0.9479, macro-F1 = 0.9477

Interpretation:
- This large jump versus 3-class performance strongly supports the label-overlap hypothesis (especially the Medium boundary).

### 2.3 Window length sweep (2 s / 4 s / 6 s)
- 6 s: val macro-F1 = 0.9559, test macro-F1 = 0.4550, test acc = 0.5891
- 4 s: val macro-F1 = 0.9492, test macro-F1 = 0.4119, test acc = 0.5310
- 2 s: val macro-F1 = 0.8919, test macro-F1 = 0.4354, test acc = 0.5029

Best by validation macro-F1: 6 s
Best by test metrics in this sweep: 6 s

### 2.4 Targeted ablations
- DeepConvNet_tuned: accuracy = 0.5843, macro-F1 = 0.5509
- CNN_no_RNN: accuracy = 0.4429, macro-F1 = 0.4295
- CNNLSTM_small: accuracy = 0.4286, macro-F1 = 0.4109

Conclusion:
- Tuned DeepConvNet is the best 3-class result among tested deep variants and also beats the SVM baseline in this run.

## 3. Medium-Class Confusion Evidence
Medium true samples total: 695

Predicted distribution for Medium class:
- SVM: Medium->Low 524, Medium->Medium 123, Medium->High 48
- DeepConvNet: Medium->Low 523, Medium->Medium 168, Medium->High 4
- CNN-LSTM: Medium->Low 475, Medium->Medium 73, Medium->High 147
- DeepConvNet_tuned: Medium->Low 523, Medium->Medium 165, Medium->High 7
- CNNLSTM_small: Medium->Low 495, Medium->Medium 99, Medium->High 101
- CNN_no_RNN: Medium->Low 402, Medium->Medium 181, Medium->High 112

Interpretation:
- All models still struggle to isolate Medium.
- Most errors are Medium -> Low (dominant mode).

## 4. What Should Be Implemented in the Main Notebook
## 4.1 Promote tuned DeepConvNet to primary deep model
Implement these settings as default for DeepConvNet training:
- Learning rate: 3e-4
- Dropout: 0.4
- Weight decay: 5e-4
- Batch size: 32
- Epochs: up to 50 with early stopping (patience 8)
- Class-weighted cross-entropy: enabled

Reason:
- Best macro-F1 (0.5509) and best accuracy (0.5843) in the tested 3-class deep models.

## 4.2 Keep SVM as a strong baseline
Keep SVM + band-power as a benchmark model.

Reason:
- It remains competitive and helps identify whether deep models deliver meaningful gains.

## 4.3 Use 6-second windows for main experiments
Change preprocessing default window size from 2.0 s to 6.0 s in the main run configuration, while still documenting 2 s and 4 s as ablation evidence.

Reason:
- Best validation macro-F1 and best test metrics in the sweep performed here.

## 4.4 Keep binary Low-vs-High as a diagnostic section
Add a short diagnostic section (not replacing main 3-class task) that reports binary Low-vs-High performance.

Reason:
- Confirms whether pipeline quality is acceptable when overlap is reduced.

## 4.5 Deprioritize CNN-LSTM for final model claims
Do not use CNN-LSTM as the primary model in the main notebook/report conclusions.

Reason:
- Both full and simplified CNN-LSTM variants underperformed.

## 4.6 Add explicit Medium-overlap reporting
In the main notebook, add mandatory outputs:
- Per-class recall table
- Medium confusion breakdown (Medium->Low, Medium->Medium, Medium->High)
- Confusion matrices for each model

Reason:
- Supports the core explanation of failure mode with quantitative evidence.

## 5. Additional Techniques to Add Next (Optional but Recommended)
1. Label smoothing for deep models to reduce overconfidence near class boundaries.
2. Focal loss experiment to prioritize hard/ambiguous samples.
3. Mild data augmentation (temporal jitter, channel dropout) for robustness.
4. Probability calibration (temperature scaling) before final report figures.
5. Subject-wise normalization variant test (if compute budget allows).

## 6. Proposed Main Notebook Update Order
1. Update preprocessing default window length to 6 s.
2. Replace current deep baseline emphasis with tuned DeepConvNet settings.
3. Keep SVM section unchanged as baseline reference.
4. Add binary diagnostic subsection (Low vs High).
5. Add Medium-overlap analysis table/prints to evaluation section.
6. Keep CNN-LSTM as optional/appendix experiment rather than primary path.

## 7. Risks and Caveats
- The window sweep used a lightweight classifier for speed; confirm the 6 s benefit with tuned DeepConvNet and SVM end-to-end before final freezing.
- Strong binary performance does not solve 3-class grading directly; it is diagnostic evidence, not the final target.
- Medium-class overlap may partly reflect label/process design, not just model capacity.
