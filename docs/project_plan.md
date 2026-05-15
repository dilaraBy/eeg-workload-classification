# EEG Mental Workload Classification - Comprehensive Project Plan

## 1) Project Context

- Module: WMG9B7-15 Artificial Intelligence and Deep Learning
- Assessment type: Individual assignment (Part 1 Notebook + Part 2 Critical Reflection)
- Submission deadline: Monday 27 April 2026, 12:00
- AI policy level: AI Collaboration (allowed with transparent acknowledgement)

### Project theme
Classify mental workload (Low, Medium, High) from EEG signals recorded during MATB task conditions (MATB easy, MATB medium, MATB difficult).

### Core research question
Can a deep learning model classify 3-level mental workload from MATB EEG more effectively than a traditional machine learning baseline under cross-session evaluation?

## 2) Problem Definition and Scope

### Problem statement
Operators in safety-critical environments can suffer performance degradation when mental workload rises. A reliable EEG-based workload classifier could support passive monitoring systems and adaptive decision support.

### In scope
- MATB EEG only (easy, med, diff)
- 3-class classification (Low/Medium/High)
- Cross-session evaluation (train on earlier sessions, test on later session)
- Baseline ML model + at least one deep model
- Reproducible notebook and critical reflection

### Out of scope
- Transportation or mobility application framing
- Reuse of dissertation, undergraduate, or group-assessment project/dataset
- Non-reproducible local-only execution for marker run-all workflow

## 3) Assessment Requirements Converted Into Execution Rules

## 3.1 Mandatory notebook rules
- Main notebook must run end-to-end with Run All.
- Include a README cell at the top with:
  - project goal,
  - dataset source and license,
  - exact run steps,
  - expected runtime,
  - outputs generated.
- Include both markdown explanation and code comments where needed.
- Keep runtime practical for marking environment; if full training is long, use reduced epochs in notebook and clearly state full-training setting used for reported results.

## 3.2 Data access and reproducibility rules
- Marker guidance states dataset should be fetched programmatically in notebook from a remote source.
- Avoid dependence on personal absolute local paths.
- Add robust checks and clear fallback messages if remote download fails.

## 3.3 Reflection/report rules
- Emphasis is critical reflection, not just implementation summary.
- Conceptual ML vs DL comparison is acceptable (full ML implementation still recommended for stronger evidence).
- Include references/citations throughout. Missing required dataset citation can cause major penalty risk.
- Include AI usage acknowledgement appendix (tool, purpose, and adaptation/ownership evidence).

## 3.4 Submission packaging constraints
- Accepted formats include ipynb, py, pdf/docx, txt.
- Max 5 files total in submission package.
- If multiple notebooks exist, clearly identify one primary notebook for marker Run All.

## 4) Current Workspace Understanding

### Primary folder
- eeg-workload-classification/
  - dataset/ (BIDS-like subject/session structure with EEGLAB .set/.fdt files)
  - Individual Assignment/ (brief + marking rubric)
  - knowledge-files/ (Q&A, models references, draft structure)

### Observed EEG file pattern (example)
- dataset/sub-01/sub-01/ses-S1/eeg/MATBeasy.set
- dataset/sub-01/sub-01/ses-S1/eeg/MATBmed.set
- dataset/sub-01/sub-01/ses-S1/eeg/MATBdiff.set

### Companion starter implementation available
- eeg-workload-classification-start/ contains reusable code and experiments:
  - src/preprocessing.py, src/features.py, src/models.py, src/evaluate.py
  - notebooks and result csv files
  - existing measured results around chance-level to modest gains
- Existing outputs indicate class imbalance/challenge (especially Medium class) and cross-session drift sensitivity.

## 5) Target Deliverables

## 5.1 Part 1 technical deliverables
- 1 main reproducible notebook (submission notebook)
- Supporting scripts only if necessary and within file-count limits
- Model outputs:
  - metrics table,
  - confusion matrices,
  - training curves,
  - brief interpretability analysis (for stronger LO2 evidence)

## 5.2 Part 2 reflection deliverables
- Structured critical reflection report mapped to LO1/LO2/LO3
- AI usage appendix with transparent disclosure
- Reference list (academic + selected industry/commercial sources)

## 5.3 Optional strengtheners
- Ablation table (with/without alignment/augmentation)
- Error analysis by class
- Concise discussion of deployment risks and safeguards

## 6) Technical Workstreams

## Workstream A - Data ingestion and labeling
Objective: produce reliable feature/epoch arrays and labels for MATB easy/med/diff.

Tasks:
1. Implement data loader for BIDS subject/session/eeg structure.
2. Restrict to MATB conditions only.
3. Map labels consistently:
   - MATBeasy -> 0 (Low)
   - MATBmed -> 1 (Medium)
   - MATBdiff -> 2 (High)
4. Log per-subject/per-session class counts before training.

Acceptance criteria:
- Deterministic loaded sample counts.
- No accidental inclusion of non-MATB tasks.

## Workstream B - Preprocessing
Objective: improve signal quality while preserving generalization.

Baseline preprocessing:
- Bandpass (for example 1-40 Hz)
- Artifact rejection thresholding
- Channel/time normalization

Improved preprocessing options:
- Euclidean Alignment for session/subject covariance shift
- Sliding-window augmentation
- Optional ICA cleanup if time allows

Acceptance criteria:
- Pre/post preprocessing sample counts recorded.
- No train-test leakage in normalization statistics.

## Workstream C - Model baselines
Objective: produce credible ML vs DL comparison.

ML baseline:
- Band-power feature extraction (theta/alpha/beta)
- StandardScaler + SVM (RBF)

Deep models:
- EEGNet (primary DL baseline)
- CNN-LSTM (secondary DL candidate)
- Optional: LaBraM fine-tune as stretch if compute/time allow

Acceptance criteria:
- Same split protocol used for fair comparison.
- Hyperparameters and seeds documented.

## Workstream D - Evaluation and analysis
Objective: produce evidence suitable for critical reflection.

Metrics:
- Accuracy
- Macro F1
- Per-class recall/precision
- Confusion matrix

Evaluation design:
- Cross-session protocol as primary
- Optional grouped subject-wise checks for robustness

Analysis outputs:
- Failure-mode commentary (especially Medium class confusion)
- Comparison table: ML vs DL trade-offs
- Short reproducibility note: hardware/runtime/packages

Acceptance criteria:
- All final tables/figures generated automatically by notebook.

## Workstream E - Report integration
Objective: directly align narrative with rubric language.

LO1:
- Explain ML vs DL differences and why DL is justified for EEG spatiotemporal patterns.

LO2:
- Critically evaluate implementation decisions, limitations, implications (ethical/social/environmental).

LO3:
- Discuss future trends (for example EEG foundation models, self-supervised pretraining, transformer approaches, multimodal fusion, explainability).

Acceptance criteria:
- Every major claim supported by evidence (results or citation).

## 7) Detailed Timeline (From 23 Apr 2026 to Submission)

## Phase 0 - Planning and compliance lock (23 Apr)
- Finalize project scope and submission constraints.
- Freeze dataset/task mapping and split protocol.
- Prepare notebook skeleton with README and section headers.

Done criteria:
- One-page execution checklist exists and is agreed.

## Phase 1 - Data and preprocessing finalization (23-24 Apr)
- Implement robust data loading for MATB files.
- Produce baseline and improved preprocessing pipelines.
- Save intermediate arrays only if needed for speed.

Done criteria:
- Reproducible data pipeline runs with clear counts and no leakage.

## Phase 2 - Modeling and evaluation (24-25 Apr)
- Train/evaluate SVM baseline.
- Train/evaluate EEGNet.
- If stable and time remains, train/evaluate CNN-LSTM.
- Generate all core plots and metrics table.

Done criteria:
- At least one ML and one DL result set with confusion matrices.

## Phase 3 - Reflection drafting and evidence mapping (25-26 Apr)
- Draft LO1/LO2/LO3 sections using actual results.
- Add critical limitations and justified improvements.
- Add implications and AI trend discussion linked to this exact problem.

Done criteria:
- Report draft complete with citations and coherent argument flow.

## Phase 4 - Reproducibility hardening and submission pack (26-27 Apr)
- Run notebook from clean kernel using Run All.
- Verify runtime and outputs.
- Finalize AI usage appendix and file packaging (<=5 files).

Done criteria:
- Submission package ready before deadline buffer.

## 8) Risk Register and Mitigations

1. Risk: Notebook fails in marker environment.
- Mitigation: use programmatic dataset retrieval, pinned install cell, clean-kernel Run All test.

2. Risk: Runtime too long for marking.
- Mitigation: reproducible quick-run mode with reduced epochs and explicit note of full-run settings.

3. Risk: Low accuracy around chance level.
- Mitigation: emphasize critical analysis quality, class-wise diagnostics, protocol difficulty, and justified next-step improvements.

4. Risk: Medium class consistently under-detected.
- Mitigation: class weighting, augmentation, threshold/error analysis, and explicit interpretation in report.

5. Risk: Data leakage inflates results.
- Mitigation: subject/session-aware splits, train-only normalization fit, code assertions.

6. Risk: Missing citation/licensing compliance.
- Mitigation: add dataset citation early in notebook and report; verify bibliography completeness.

7. Risk: AI misuse policy non-compliance.
- Mitigation: include transparent appendix documenting AI assistance and human adaptation decisions.

## 9) Definition of Done

The project is complete when all items below are true:

- Main notebook runs end-to-end from clean kernel with clear README instructions.
- Contains data loading, preprocessing, model training, evaluation, and visual outputs.
- Includes one traditional ML baseline and one DL model comparison.
- Reflection/report demonstrates critical evaluation and aligns explicitly to LO1, LO2, LO3.
- References and dataset citation are complete and valid.
- AI usage appendix is included and compliant.
- Submission files are within count/format constraints.

## 10) Submission-Ready Checklist

- [ ] Main notebook chosen and clearly named
- [ ] README cell at top with Run All instructions
- [ ] Dataset source, DOI/license, and citation included
- [ ] Reproducibility install/setup cells included
- [ ] Metrics and confusion matrices generated in notebook
- [ ] ML vs DL comparison included
- [ ] Critical reflection report finalized (word count within limits)
- [ ] AI usage acknowledgement appendix included
- [ ] Final clean-kernel run completed without errors
- [ ] Final file bundle within allowed number and formats

## 11) Recommended Immediate Next Actions

1. Create final submission notebook skeleton in eeg-workload-classification/ with all required sections and README.
2. Port the best-performing reusable code modules from eeg-workload-classification-start/ into that notebook/script flow.
3. Run one full baseline experiment first (SVM + EEGNet) to lock evidence for report writing.
4. Draft reflection in parallel while experiments run, using LO-mapped headings from the start.
