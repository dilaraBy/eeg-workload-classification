# WM9B7 — Final Push: CNN-LSTM Improvements + Distinction Gaps + Applied Demo

**Current best:** CNN-LSTM acc=0.5989, macro-F1=0.5768, Medium-F1=0.5128
**Goal:** Squeeze more from the *existing* CNN-LSTM without adding new model architectures, fill rubric gaps for distinction (70+), and add an applied demonstration component.

This document has three parts:
1. **Part A** — Literature-backed improvements to your CNN-LSTM
2. **Part B** — Rubric gap analysis: what's missing for distinction
3. **Part C** — Applied demonstration / visualisation ideas

---

# Part A — Improving the CNN-LSTM (literature review)

These are changes to your *existing* multi-scale CNN-LSTM architecture and training pipeline. No new models. Ordered by expected impact and implementation ease.

---

## A1. Mixup Augmentation (highest priority)

**What it is:** During training, randomly blend two EEG windows and their labels with a mixing coefficient λ ~ Beta(α, α). The model trains on the interpolated sample with a soft label.

**Why the literature says it works for your problem:**
- Zhang et al. (2018, ICLR) showed mixup smooths decision boundaries between adjacent classes — exactly your Medium-vs-Low bleed.
- EEG-specific applications: Mohsenvand et al. (2021) applied mixup to self-supervised EEG pretraining and reported improved sample efficiency. Zhou et al. (2023) proposed EEG-Mixup, mixing within the same subject's data to preserve physiological identity while augmenting across classes. The Frontiers 2024 few-shot EEG survey specifically recommends mixup for data augmentation in mental workload detection (MWD).
- The AMDA (Adversarial + Mixup Data Augmentation) framework in seizure detection (PMC 2024) applied mixup alongside adversarial perturbations to a CNN-GRU (architecturally similar to your CNN-LSTM) and reported significant generalisation improvements.

**Why it fits your specific problem:**
Your confusion matrix shows Medium→Low is the dominant error. Mixup forces the network to learn gradual transitions between class regions rather than hard boundaries. With α=0.2, most mixed samples are close to one class but with soft labels — this is ideal for the ambiguous Medium region where "Medium-ish" and "Low-ish" windows genuinely overlap.

**Implementation (add to your existing training loop):**
```python
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam

# In training loop:
x_mix, y_a, y_b, lam = mixup_batch(X_batch, y_batch, alpha=0.2)
logits = model(x_mix)
loss = lam * focal_loss(logits, y_a) + (1 - lam) * focal_loss(logits, y_b)
```

**Expected gain:** +2–4 pp macro-F1, with the largest gain on Medium-F1.

**Key references:**
- Zhang, H., Cisse, M., Dauphin, Y.N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. ICLR.
- Mohsenvand, M.N., Izadi, M.R., & Maes, P. (2021). Contrastive representation learning for electroencephalogram classification. Machine Learning for Health (ML4H) at NeurIPS.
- Zhou, Y. et al. (2023). EEGmixup augmentation for cross-domain EEG classification.

---

## A2. Adaptive Batch Normalisation at Test Time (AdaBN)

**What it is:** After training, do one unlabelled forward pass over all S3 test windows with BatchNorm layers in train mode (collecting running statistics), then switch to eval mode. No weight updates. No labels used.

**Why this is relevant:** Your model uses InstanceNorm, which is per-sample and session-agnostic — so AdaBN will NOT work on your current architecture directly. However, InstanceNorm was chosen *because* BatchNorm running stats leak session identity. The literature shows a better compromise:

**Adapt to use:** Replace InstanceNorm with BatchNorm in the CNN-LSTM, BUT apply AdaBN at test time. This gets you the training-time benefits of BatchNorm (smoother gradients, faster convergence) while using AdaBN to recalibrate the statistics to S3 at test time — best of both worlds.

**Literature:**
- Li, Y. et al. (2018). Adaptive Batch Normalization for practical domain adaptation. Pattern Recognition, 80, 109–117. — original AdaBN paper.
- Pérez-García et al. (2024). Calibration-free online test-time adaptation for EEG motor imagery decoding. — applied AdaBN + entropy minimisation to EEG cross-subject, achieved SOTA without any calibration data. They specifically found that adaptive BN statistics recalculation on the test stream was the single most effective TTA technique for EEG.
- The Frontiers 2025 comprehensive DL-EEG review recommends the "practical recipe": (i) self-supervised pretraining, (ii) lightweight subject-wise finetuning, (iii) adversarial/contrastive alignment — but notes AdaBN as the simplest drop-in for (iii) when compute is limited.

**Implementation:**
```python
# After training, before evaluation:
model.train()  # puts BN into stat-collection mode
with torch.no_grad():
    for x_batch, _ in test_loader:
        _ = model(x_batch.to(device))  # BN absorbs S3 statistics
model.eval()
# Now evaluate — BN running_mean/var reflect S3 distribution
```

**Expected gain:** +2–4 pp if you switch InstanceNorm→BatchNorm+AdaBN. If you keep InstanceNorm, this step has no effect — skip it.

**Key references:**
- Li, Y. et al. (2018). Adaptive Batch Normalization for practical domain adaptation. Pattern Recognition.
- Pérez-García, F. et al. (2024). Calibration-free online test-time adaptation for EEG motor imagery decoding. arXiv:2311.18520.

---

## A3. Domain-Adversarial Training (DANN) on Your CNN-LSTM Backbone

**What it is:** Add a second head (domain classifier) to your CNN-LSTM that tries to predict whether a sample is from the training sessions (S1+S2) or the test session (S3). A gradient reversal layer forces the backbone to learn features that are session-invariant.

**Why this is the right technique for your exact problem:**
Your report §6.3 identifies that 8/28 subjects are at or below chance — the cross-session shift is the bottleneck. DANN directly attacks this. The literature is extremely consistent here:
- Ganin & Lempitsky (2015, ICML) — original DANN. Standard technique for unsupervised domain adaptation.
- Liu et al. (2022) applied DANN to cross-subject EEG motor imagery classification with CNN backbones, reporting significant improvements in domain-invariant feature learning.
- Li et al. (2024) applied transformer-based domain adaptation specifically for EEG cognitive workload estimation — the exact task class you're working on.
- Jin et al. (2024, Biomed. Signal Process. Control) proposed Transfer DS-CNN for cross-session mental workload recognition using SDAE/ASFM domain adaptation approaches.
- The DSP-EmotionNet paper (Frontiers 2024) combined DANN with CNN feature extractors for cross-subject EEG classification and outperformed non-adversarial baselines across all metrics.

**Critical point:** You use S3 windows without their class labels — only a binary domain label (train=0 / test=1). This is not test-set leakage. Document this clearly.

**Implementation — add to your existing CNN-LSTM:**
```python
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

# After the BiLSTM + attention (before the classifier head):
# features = GAP output (128-dim)
#   ├── classifier_head → 3-class workload
#   └── GradientReversal → domain_head → 2-class (train/test session)
```

Train with: `total_loss = class_loss + 0.3 * domain_loss`

**Expected gain:** +3–7 pp macro-F1. This is the single largest gain reported in the cross-session EEG DA literature.

**Key references:**
- Ganin, Y. & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. ICML.
- Jin, L. et al. (2024). Identifying stable EEG patterns over time for mental workload recognition using transfer DS-CNN. Biomed. Signal Process. Control, 89, 105662.
- Li et al. (2024). Transformer-based domain adaptation for EEG cognitive workload (cited in Frontiers 2025 review).

---

## A4. RSME Auxiliary Regression Head (multi-task learning)

**What it is:** Add a regression head alongside the classifier that predicts the subjective mental effort score (RSME: Low≈35, Medium≈55, High≈95). Trained jointly with a small MSE loss weight.

**Why it helps:** The classifier sees 3 discrete labels with a hard boundary. The regression head sees a continuous 35→55→95 gradient. For the Medium class (RSME≈55), the regression signal provides smoother gradients than the discrete 0/1/2 label — the network gets "warmer/colder" feedback rather than binary right/wrong. This is particularly valuable because your §9 notes the RSME easy→medium gap is smaller than medium→hard, which matches the classification difficulty.

**Note:** RSME scores are *not* used as features at inference. They are training-only auxiliary targets. At test time, only the classification head is used. This means the system remains a pure EEG-based classifier — not dependent on self-report data for deployment.

**Literature support:**
- Multi-task learning for EEG is well established. Shao et al. (2023) combined Bi-LSTM and CNN in a multi-task framework for mental workload classification using spatial + time-frequency features.
- Auxiliary regression heads in cognitive state classification are advocated in the ACM TOCH 2025 review as a way to regularise intermediate-class boundaries when the physiological variable is ordinal.

**Implementation:**
```python
# After GAP (128-dim features):
self.regressor = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
# RSME targets: y_rsme = {0: 35.0, 1: 55.0, 2: 95.0}[class_label]
# Loss: total = focal_loss + 0.1 * MSE(reg_pred, rsme_target)
```

**Expected gain:** +1–3 pp on Medium-F1 specifically.

---

## A5. Stochastic Weight Averaging (SWA)

**What it is:** Average the model weights from the last N epochs of training instead of taking the final checkpoint. Produces a flatter minimum that generalises better.

**Why it fits:** Your CNN-LSTM trains for 50 epochs with cosine annealing, ending at LR≈0. The final few epochs explore a flat region of the loss landscape — averaging those weights produces a more robust model than any single checkpoint. This is a pure training trick that requires zero architecture changes.

**Literature:**
- Izmailov et al. (2018). Averaging weights leads to wider optima and better generalization. UAI.
- PyTorch has built-in `torch.optim.swa_utils.AveragedModel`.

**Implementation:**
```python
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model)
# After epoch 35 (when cosine LR is low):
swa_model.update_parameters(model)
# At end of training:
torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

**Expected gain:** +1–2 pp accuracy with zero downside risk.

---

## A6. Ordinal-Aware Loss (replaces or supplements focal loss)

**What it is:** Standard cross-entropy treats Low→High misclassification the same as Low→Medium. But workload is ordinal (Low < Medium < High). An ordinal loss penalises predictions that are far from the true rank more than those that are close.

**Why it helps:** Your Medium→Low error is dominant, but it is a *one-rank* error. Medium→High is also one-rank. Low→High is a *two-rank* error and should be penalised more heavily. This naturally pushes the model to at least get the ordering right.

**Implementation — simple rank penalty added to focal loss:**
```python
# After computing focal loss:
rank_penalty = torch.abs(preds.argmax(1).float() - targets.float()).mean()
total_loss = focal_loss + 0.05 * rank_penalty
```

Or use the CORN framework (Cao et al., 2020) for proper ordinal regression with neural networks.

**Expected gain:** +1–2 pp on Medium-F1.

**Key reference:**
- Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks. Pattern Recognition Letters.

---

## Summary — what to implement and in what order

| Priority | Change | Touches | Expected Δ macro-F1 |
|---|---|---|---|
| 1 | Mixup (α=0.2) | Training loop only | +2–4 pp |
| 2 | DANN domain head | Add 1 head + GRL | +3–7 pp |
| 3 | AdaBN (if switch to BN) | Norm layers + test-time | +2–4 pp |
| 4 | RSME auxiliary head | Add 1 head + MSE loss | +1–3 pp |
| 5 | SWA | Post-training averaging | +1–2 pp |
| 6 | Ordinal rank penalty | Loss function | +1–2 pp |

**Do items 1+2 first.** They are independent and stack. If your CNN-LSTM moves from macro-F1 0.577 to ~0.63–0.65, that is a strong, defensible improvement.

---

# Part B — Rubric Gap Analysis for Distinction (70–79)

I re-read the assessment brief and marking rubric line by line against your current report. Here is what the 70+ band requires that you are currently missing or underserving.

---

## B1. LO1 — ML vs DL Comparison (currently: ~65 band)

**What the 70+ rubric says:** *"Strong, critical comparison that reflects sound theoretical and practical knowledge. Justification for the chosen DL approach is well-integrated with the problem context."*

**What you have:** Your SVM baseline collapsed (macro-F1=0.21) due to the CORAL misapplication. You do not have a functioning ML baseline to compare against in the final results. The reflection (§8) acknowledges this but the report currently has no *working* ML-vs-DL comparison.

**What you need to fix:**
1. **Rerun SVM without CORAL** (or with symmetric CORAL). Get a clean SVM number on 8s windows. Even if it is ~0.48–0.50, it gives you a real comparison row.
2. **Add a structured comparison paragraph** in the reflection: *why* the CNN-LSTM outperforms SVM — end-to-end temporal feature learning from raw EEG vs hand-crafted band-power features; ability to capture cross-frequency interactions; temporal attention selecting discriminative time segments that fixed windows cannot. Cite Lawhern et al. (2018), Schirrmeister et al. (2017) for the established argument.
3. **Discuss where ML still wins:** SVM is faster to train, more interpretable (feature importances map to known frequency bands), lower compute cost. Cite the Frontiers 2025 neuroergonomics review which notes that traditional ML remains relevant for rapid prototyping and when data volume is small.

---

## B2. LO2 — Evaluate DL Solution + Implications (currently: ~65 band)

**What the 70+ rubric says:** *"Strong implementation with thoughtful design and use of tools, and well-documented. Evaluation is comprehensive and includes interpretability. The reflection presents a clear, critical discussion of implications."*

**What you're missing:**

### B2a. Interpretability beyond Grad-CAM
Your Grad-CAM returned a "flat/uniform" saliency pattern — you correctly note this is physiologically plausible, but it gives zero insight into *which channels or frequency bands* the model relies on. The rubric wants interpretability that connects to neuroscience.

**Add:** A **channel-importance** analysis. Zero out one channel at a time (channel ablation / perturbation importance), run inference, measure the macro-F1 drop. Plot the top-10 most important channels on a scalp topography map. If frontal channels (Fz, FCz, F3, F4) dominate for the High class, this aligns with frontal-theta workload literature (So et al., 2017; Pergher et al., 2019). If parietal channels (Pz, P3, P4) dominate for Low, this aligns with alpha-suppression during disengagement. Either way, you get a figure that tells a neuroscience story.

This is worth significant marks: the rubric for 70+ explicitly says "includes interpretability."

### B2b. Ethical / environmental / societal implications
Your report has *none of this*. The rubric for LO2 at 70+ says: *"The reflection presents a clear, critical discussion of implications."* At 80+: *"Ethical, ecological, and societal implications are critically analysed with maturity."*

**Add a section (≈300 words) covering:**

- **Individual impact:** False negatives in safety-critical contexts (surgeon told they are not overloaded when they are → error risk). Consent and autonomy — continuous cognitive monitoring raises surveillance concerns.
- **Organisational impact:** Employers using workload classifiers to monitor staff productivity; liability if the system fails; potential for discriminatory effects if the model performs poorly for certain demographic groups (your per-subject analysis already shows this).
- **Environmental impact:** Training compute footprint — your model is small (~35K params, trivial). But scaling to continuous real-time monitoring across an organisation requires edge inference hardware. Cite Strubell et al. (2019) or a more recent equivalent on energy costs of DL.
- **Societal / regulatory:** The EU AI Act (2024) classifies biometric categorisation systems (including emotion/cognitive state inference from physiological signals) as high-risk. EEG workload classifiers would likely fall under this. Cite Ienca & Andorno (2017) on neuro-rights.

### B2c. AI assistant disclosure
The brief explicitly requires you to *"discuss how you used AI assistants (if applicable) to support your implementation and how you expanded or adapted the generated output."* Your report does not mention this. Add a short paragraph (~100 words) describing what you used, what you changed, and one concrete example of a suggestion you overrode or adapted.

---

## B3. LO3 — Emerging Trends (currently: ~60 band)

**What the 70+ rubric says:** *"Strong awareness of current AI research and innovation. Trends are contextualised, with clear links to the problem domain and project future."*

**What you have:** Your §7 briefly mentions transformers and behavioural data but doesn't engage with the current SOTA trends.

**What you need to add (~400 words):**

Pick 2–3 of these and connect each one *directly* to your problem and your CNN-LSTM:

1. **EEG Foundation Models (LaBraM, CBraMod, BENDR):** Pretrained on 2,500+ hours of diverse EEG, these models learn general neural representations via masked token reconstruction. AdaBrain-Bench (2025) reported LaBraM achieving 85.83% balanced accuracy on EEGMAT (workload classification) — vs 73.89% for the best from-scratch supervised model. Applied to your problem, fine-tuning a pretrained LaBraM backbone instead of training CNN-LSTM from scratch could solve the cross-session shift by leveraging general EEG representations that are inherently more session-invariant. Cite: Jiang et al. (2024, ICLR); Kostas et al. (2021).

2. **Test-time adaptation (TTA) / Online adaptation:** Pérez-García et al. (2024) showed that simple techniques (AdaBN, entropy minimisation) applied at inference time — without any labelled calibration data — achieve SOTA cross-subject EEG decoding. This directly addresses your §6.3 finding that calibration would be the "single largest improvement" — TTA achieves partial calibration *without* requiring labelled S3 windows. This makes real-time deployment feasible.

3. **Multimodal fusion (EEG + ECG + eye-tracking):** COG-BCI also includes ECG data. Heart-rate variability (HRV) is a well-established workload correlate that is less affected by session-to-session electrode shifts than EEG. Fusing EEG with HRV features could stabilise the 8 subjects currently at chance. Cite: Mehmood et al. (2023) on multimodal integration for cognitive fatigue.

---

## B4. Missing structural elements

| Gap | What to add | Word cost |
|---|---|---|
| Abstract / summary | A 100-word summary at the top of the reflection | ~100 |
| AI assistant disclosure | How you used AI tools + what you adapted | ~100 |
| Ethics / implications section | Individual, org, environmental, societal | ~300 |
| Future trends section (LO3) | 2–3 trends linked to your problem | ~350 |
| Working SVM baseline | Rerun SVM without CORAL; report clean number | 0 (notebook) |
| Reference list expansion | Add ~8–10 more references (currently 11, need 20+) | 0 |

**Total additional words: ~850.** Your current report structure leaves room since the reflection is separate from the notebook. Check your current word count — you have a 2,800-word limit.

---

# Part C — Applied Demonstration / Visualisation

The brief says the notebook should contain "appropriate methods, e.g., data preprocessing, model selection, training, and evaluation, among others." It doesn't require a demo app. But adding an applied component demonstrates real-world relevance (LO2 implications) and shows initiative (80+ rubric: "originality of thought").

Here are three options, from easiest to most impressive:

---

## C1. Interactive Workload Monitor Dashboard (React artifact — easiest, in-notebook)

Create a single-file React or HTML artifact that simulates what a deployed system would look like. It loads pre-computed model predictions for each subject across the 8-second windows and displays:

- A **timeline view**: for a selected subject, show the predicted workload level (Low/Medium/High) across the session, colour-coded, alongside the ground truth. Shows where the model succeeds and where it fails.
- A **topographic scalp map**: show the channel-importance heatmap from the perturbation analysis (B2a).
- A **per-subject accuracy bar chart**: the 28 subjects ranked by accuracy, colour-coded by above/below chance.
- A **confusion matrix** that updates when you select a subject.

This is a **post-hoc analysis dashboard**, not a real-time system (you don't have streaming EEG). But it visualises the results in a way that a clinician or operator would interact with.

**Implementation:** Export predictions to a JSON/CSV in the notebook, then build a React artifact that reads it. No server needed.

---

## C2. Simulated Real-Time Workload Classifier (React artifact — medium)

Build an interactive artifact that:
1. Lets the user pick a subject and a session.
2. "Streams" 8-second windows one at a time (simulated, from pre-extracted test data).
3. Shows a sliding-window EEG trace (just 3 channels: Fz, Cz, Pz) for the current window.
4. Displays the model's predicted workload level with a confidence gauge (softmax probabilities).
5. Shows a running accuracy counter.

This demonstrates what a *deployed* BCI workload monitor would look like in a control room or surgical suite. Even though it's replaying stored data, it makes the real-world application tangible.

---

## C3. Notebook-Only Visualisation Cells (simplest, no artifact)

If you don't want a separate app, add these visualisation cells to the notebook:

1. **EEG trace + prediction overlay**: For 3 example subjects (one good, one medium, one bad), plot 60 seconds of raw EEG (Fz channel) with the 8-second windows colour-coded by predicted class. Overlay ground truth as a bar above.

2. **Training curves**: Loss and validation macro-F1 across epochs for all models, on the same axes. Shows that CNN-LSTM trains most stably.

3. **Scalp topography of channel importance**: MNE's `plot_topomap()` with the perturbation importance values.

4. **Confusion matrix heatmap**: seaborn heatmap with annotations.

5. **Band-power spectra per class**: PSD averaged across subjects for each class, with theta/alpha/beta bands shaded. Shows the physiological basis for classification.

6. **Per-subject accuracy waterfall**: Horizontal bar chart, all 28 subjects, sorted by accuracy, with a dashed line at chance (33.3%).

These are standard figures that examiners expect and the rubric rewards under "well-documented" and "clear visualisations."

---

## My recommendation

Do **C1 or C3** depending on time:
- If you have 2–3 extra hours, do **C1** — a dashboard artifact in the notebook. It's the kind of thing that gets mentioned in examiner comments ("excellent applied demonstration").
- If you're tight on time, do **C3** — add 4–5 strong visualisation cells. These are expected at distinction level anyway.

Do NOT spend time on C2 unless everything else is done. The simulated streaming is impressive but adds no marks beyond C1.

---

# Final checklist before submission

- [ ] SVM baseline rerun without CORAL (clean ML comparison for LO1)
- [ ] Mixup added to CNN-LSTM training loop
- [ ] DANN domain head added (if time allows — this is the biggest single gain)
- [ ] Channel-importance perturbation analysis + topomap figure
- [ ] Ethics/implications section in reflection (individual, org, environment, society, EU AI Act)
- [ ] Future trends section with 2–3 cited trends linked to your problem
- [ ] AI assistant disclosure paragraph
- [ ] Reference list expanded to 20+ (currently 11)
- [ ] Dashboard or visualisation cells added
- [ ] Word count checked (2,800 max including tables/figures/citations)
- [ ] Notebook README cell present with how-to-run instructions
- [ ] All outputs visible (do not clear cells before submission)
