# WM9B7 AIDL Individual Assessment — Project Plan

**Project title:** EEG-based Mental Workload Classification using Deep Learning on the COG-BCI MATB-II dataset
**Module:** WMG9B7-15 — Artificial Intelligence & Deep Learning
**Weighting:** 70% (individual)
**Submission deadline:** Monday 27 April 2026, 12:00
**Word limit (reflection):** 2,800 words (+10% allowed)
**Deliverables:** (1) Reproducible Jupyter Notebook, (2) Critical Reflection report. Both uploaded to Tabula.

---

## 1. Read-this-first checklist

Before coding anything, sanity-check the rules so the project does not get capped at the pass mark on a technicality.

1. **AI policy = AI COLLABORATION.** You *may* use AI assistants to help draft text, refine code and evaluate your work, but you must critically modify everything and be able to defend every line. Keep a short log of the prompts you used and how you changed the output — this feeds directly into the "How you used AI assistants" section of the reflection.
2. **"Cannot be transportation or mobility."** The MATB-II is aviation-flavoured (it is a NASA multi-attribute task battery). This is a real risk. You are safe if you frame the problem as **"EEG-based cognitive/mental workload monitoring for safety-critical human–computer interaction"** with example domains such as *surgery, air-traffic control, power-plant operators, adaptive training, neuroergonomics, healthcare monitoring*. Do **not** frame it as "driver monitoring," "pilot monitoring for flight," or anything automotive/aviation operational. State this scoping decision explicitly in the notebook's problem-definition cell and in the reflection.
3. **Get dataset approval** from Dr Leonardo Alves Dias before you commit to it — the brief says "subject to approval."
4. **LOs assessed: LO1 (ML vs DL), LO2 (evaluate DL solution + implications), LO3 (emerging trends).** Every section of the reflection must visibly map to one of these three. Do not write a generic report.
5. **Word count includes:** quotes, tables, figures, footnotes, citations, titles, abstracts. **Excludes:** references, appendices, ToC. Put anything marginal (e.g. full hyper-parameter tables, long code listings) in an appendix.
6. **Do not reuse this work in your dissertation** (explicit brief instruction).
7. **Late penalty:** 5 marks per working day. Submit a day early with buffer.

---

## 2. The dataset — what you actually have

**Source:** COG-BCI database v4 — Zenodo record 7413650 (Hinss et al., 2022). CC-BY 4.0, so free to use with attribution.

**Scale:** 29 participants × 3 sessions × 4 tasks (MATB, N-Back, PVT, Flanker). ~31.7 GB total. For this project **use only the MATB task** to keep scope manageable.

**Hardware:** 64-channel ActiCap (Ag-AgCl active electrodes), ActiCHamp amplifier, 500 Hz sampling rate. EEG data is in **BIDS format** (`.set`/`.fdt` or similar, loadable with MNE-Python).

**MATB-II task structure:**
- Four simultaneous subtasks: System Monitoring (SYSMON), Tracking (TRACK), Resource Management (RESMAN), Communications (COMM).
- **Three difficulty levels per session: Easy, Medium, Difficult** — each block is ~5 minutes. These are your **class labels**.
- Behavioural data is a MATLAB `.mat` structure with substructures:
  - `TRACK` — 2 cols (X, Y cursor coords, 2 Hz)
  - `SYSMON` — alarm onset + reaction time
  - `RESMAN` — fuel levels in reservoirs (1 Hz)
  - plus COMM events
- Subjective questionnaires: **RSME** (Rating Scale Mental Effort) and **KSS** (Karolinska Sleepiness Scale) in top-level .txt files. These are ground-truth-ish subjective workload labels you can use as a secondary regression target or sanity check.

**The problem.** Given a multi-channel EEG epoch recorded while a participant is performing the MATB, predict which of the three workload levels (Easy / Medium / Difficult) they are experiencing. This is a classic 3-class **passive BCI** problem and has a documented baseline (MDM-Riemannian classifier ~65% in the original paper — your target is to match or beat this with deep learning).

---

## 3. Problem framing (for the notebook and reflection)

Use this wording (or similar) so you are safely clear of the transport/mobility exclusion:

> "EEG-based deep-learning classification of mental workload during multi-attribute cognitive tasks, with application to adaptive human–computer interaction in safety-critical operator environments — for example, surgical training, control-room operations, cognitive rehabilitation, and neuroergonomic research. Accurate, real-time workload monitoring enables interfaces that adapt their complexity to prevent operator overload or disengagement."

Real-world relevance (cite these in the reflection):
- Medical: monitoring surgeon fatigue during long procedures
- Education: adaptive tutoring systems that adjust difficulty to learner workload
- Neuroergonomics: objective workload measurement replacing self-report (NASA-TLX, RSME) which is retrospective and biased
- Human-in-the-loop AI: knowing when the human is overloaded so the AI can take over low-level tasks

---

## 4. Scope decision — which subset to actually use

Using all 29 × 3 × 3 conditions × 64 channels × 500 Hz × 5 min = far too much for a 15-credit module and a laptop. Recommended de-scoping:

| Decision | Recommendation | Rationale |
|---|---|---|
| Participants | All 29 (or minimum 15 if compute-limited) | Subject variability is the story |
| Sessions | Session 1 only first; extend to S1+S2+S3 for a transfer-learning stretch goal | Reduces compute 3× |
| Task | MATB only | Scope |
| Channels | All 64 (keep full montage) | DL models benefit from spatial info |
| Downsampling | 500 Hz → 128 Hz | Standard for EEG DL; ~4× speed |
| Epoching | 2 s non-overlapping windows inside each 5-min block, labelled by block difficulty | Gives ~150 epochs per block → ~1,350 epochs per subject — enough |
| Train/val/test split | **Leave-One-Subject-Out (LOSO)** cross-validation OR stratified subject-disjoint split | Never split within-subject — that leaks |

**Critical pitfall to avoid:** random 80/20 splits across all epochs will leak because epochs from the same subject end up in train and test. LOSO is the gold standard for EEG.

---

## 5. Notebook structure — cell by cell

The notebook is the single most important deliverable. Build it in this order so it reads top-to-bottom as a clean report.

### Cell 1 — README (markdown)
- Project title, author, module code
- One-paragraph problem statement (use the framing from §3)
- **How to run:** Python version (3.10+), `pip install -r requirements.txt`, path to downloaded COG-BCI data, expected run-time, GPU vs CPU note
- Dataset license + citation (Hinss et al., 2022, CC-BY 4.0)
- AI assistance disclosure (one sentence: what you used, for what)
- Ethical note: data is anonymised, obtained under CER 2021-342 ethical approval

### Cell 2 — Imports, seeds, config
```python
import numpy as np, pandas as pd, mne, torch, random
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
# config dict: paths, sfreq, epoch length, batch size, etc.
```
Set seeds *everywhere* and log library versions — reproducibility is an explicit rubric criterion.

### Cell 3 — Data loading
- Load BIDS EEG with `mne-bids` / `mne.io.read_raw`
- Loop through subjects/sessions, read events from annotations, identify MATB blocks + difficulty from the event trigger list (`triggerlist.txt`)
- Sanity print: number of blocks, duration, channel names, sampling rate

### Cell 4 — Preprocessing
Standard MNE pipeline — be explicit about every decision in a markdown cell above it:
1. Set montage (standard 10-20 / ActiCap 64)
2. Band-pass filter 1–40 Hz (FIR, zero-phase)
3. Notch 50 Hz (European line noise)
4. Re-reference to common average
5. Resample to 128 Hz
6. Bad channel detection (PREP or simple variance-based), interpolate
7. ICA to remove EOG/EMG artefacts (optional but strong — document the decision)
8. Epoch into 2 s windows by block, drop epochs with amplitude > 150 µV

### Cell 5 — Label assignment + dataset summary
- Map each epoch to its {Easy, Medium, Difficult} label
- Plot class distribution (should be roughly balanced)
- Plot an example epoch from each class for three channels (Fz, Cz, Pz) — shows markers you are not training on noise

### Cell 6 — Exploratory analysis (EDA)
Cheap wins that also feed the reflection:
- Power spectral density per class averaged across subjects — expect **higher frontal theta (4–7 Hz)** and **lower parietal alpha (8–13 Hz)** in the Difficult condition. This is a literature-supported sanity check and gives you a figure for the report.
- Topographic maps of theta and alpha power per class
- Subject-level accuracy ceiling: look at RSME/KSS correlation with block difficulty — if a subject's self-report does not track difficulty, their data will be harder to classify

### Cell 7 — Train/test split
Implement LOSO CV iterator. Also a quick single-holdout split (e.g. 5 subjects out) for faster iteration during development.

### Cell 8 — Feature extraction (for the **traditional ML baseline**)
- Compute per-epoch per-channel PSD band power in delta/theta/alpha/beta/gamma
- Flatten to feature vector (64 channels × 5 bands = 320 features)
- Standardise per subject

### Cell 9 — Traditional ML baselines (for LO1)
Train and report:
1. **Logistic Regression** (simplest baseline)
2. **Random Forest**
3. **SVM with RBF kernel**
4. *Optional:* **Riemannian MDM** (`pyRiemann`) — this is the pipeline the original authors used; matching it is credible

Report accuracy, macro-F1, and confusion matrix for each under LOSO.

### Cell 10 — Deep learning models (the main contribution, for LO2)

Input tensor shape: `(batch, 1, channels=64, time=256)` (2 s × 128 Hz).

Train **at least two** of the following — running three gives you a stronger comparison section:

| Model | Why pick it | Rough params |
|---|---|---|
| **EEGNet** (Lawhern et al., 2018) | The standard compact CNN for EEG; ~2k params; fast; well-cited | Depthwise + separable convs |
| **ShallowConvNet / DeepConvNet** (Schirrmeister et al., 2017) | Stronger CNN baselines; good literature comparison | 2–5 conv blocks |
| **EEG-Transformer / EEG-Conformer** | Lets you talk about attention + emerging trends for LO3 | CNN front-end + transformer encoder |
| **1D-CNN + BiLSTM** | Easy to implement; good if you prefer PyTorch from scratch | ~3 conv + 1 LSTM |

Recommended pair: **EEGNet (as a strong light baseline) + EEG-Conformer (for the trend story)**.

Training specifics to document:
- Loss: cross-entropy (add class weights if imbalanced)
- Optimiser: AdamW, lr 1e-3, weight decay 1e-4
- LR scheduler: cosine or reduce-on-plateau
- Epochs: 100 with early stopping on val macro-F1 (patience 15)
- Batch size: 64
- Mixed precision if GPU

### Cell 11 — Evaluation (for LO2)
Report **under LOSO**:
- Accuracy, macro-F1, per-class precision/recall/F1
- Confusion matrix (per fold + aggregated)
- Per-subject accuracy bar chart — shows variance across subjects
- Training curves (loss + val metric vs epoch)
- Chance level with 95% CI (binomial) — crucial for an honest claim

### Cell 12 — Interpretability (key for a distinction)
Pick one, run it, show a figure:
- **Integrated Gradients** or **Saliency maps** showing which channels × time points drove the prediction. Expect frontal/central channels to dominate for Difficult — this cross-checks with the EDA and with the theta-band literature (Bowers et al., 2014; Ke et al., 2021).
- Alternative: **Grad-CAM** on the final conv layer averaged across classes.

This is the single highest-leverage cell for hitting the 70+ band on LO2 ("Evaluation is comprehensive and includes interpretability").

### Cell 13 — Ablation / robustness
At least one ablation — pick one you can actually run:
- Drop ICA → show how much it costs
- Reduce to 16 channels → cost vs deployability trade-off
- Subject-dependent vs subject-independent training → shows calibration cost

### Cell 14 — Conclusion + limitations
2–3 bullet markdown cell: what worked, what did not, threats to validity.

### Cell 15 — References
Use a consistent style (Harvard or APA).

---

## 6. Target results and what "good" looks like

Literature references for sanity:
- **Chance (3-class):** 33.3%; 95% CI upper bound ~40% for a few thousand epochs
- **Original COG-BCI paper (Hinss et al., 2022):** ~65% with MDM-Riemannian on MATB
- **Recent EEGNet on similar 3-class workload tasks:** 65–75% LOSO
- **Your realistic target:** 60–72% LOSO macro-F1; subject-dependent much higher (80%+) but report LOSO as primary.

If your DL model matches or slightly beats the Riemannian baseline, that is a perfectly strong, honest result. Resist the urge to inflate numbers with within-subject leakage.

---

## 7. Reflection (report) — structure and word budget

Total target: **2,700 words** (leaves 100-word buffer under the 2,800 limit).

Suggested sections with word budgets and the LO each one ticks:

| # | Section | Words | LO | What to include |
|---|---|---|---|---|
| 1 | Introduction & problem statement | 250 | — | Problem, relevance, dataset, why DL matters here |
| 2 | DL vs traditional ML comparison | 550 | **LO1** | Paradigm differences (feature engineering vs end-to-end, data scale, compute, interpretability); why DL suits raw multi-channel EEG (spatial-temporal patterns, automatic feature learning); why a Riemannian/SVM baseline is still valuable. **Direct quantitative comparison from your own results table.** |
| 3 | Design decisions | 550 | **LO2** | Model choice (EEGNet, Conformer, etc.) with citations; epoch length rationale; LOSO vs random split; preprocessing trade-offs (ICA yes/no); hyper-parameter choices — each justified, not asserted. Include one figure (training curves or architecture diagram). |
| 4 | Evaluation + interpretability | 350 | **LO2** | Metrics you used and why (macro-F1 > accuracy when imbalanced); confusion-matrix insights; what the saliency maps revealed and whether it aligns with the frontal-theta / parietal-alpha literature. |
| 5 | Use of AI assistants | 200 | — (but required) | Which assistant(s), which tasks (boilerplate, debugging, prose), what you critically changed, what you rejected. One concrete example of a suggestion you overrode. |
| 6 | Impact & implications | 400 | **LO2** | Individual (mental health, fatigue, autonomy, false negatives in safety contexts); Organisational (adaptive training, liability, staff surveillance risk); Environmental (training-compute footprint; cite one paper e.g. Strubell et al. 2019 or more recent); Societal (consent, neuro-privacy, EU AI Act risk tier for biometric inference). |
| 7 | Future trends & opportunities | 400 | **LO3** | Pick 2–3, link each directly back to your problem: *(a)* **EEG foundation models / self-supervised pretraining** (e.g. BENDR, LaBraM, BrainBERT) for low-data subject adaptation; *(b)* **Multimodal models** fusing EEG + ECG + eye-tracking (COG-BCI also has ECG!); *(c)* **Federated / on-device learning** for privacy; *(d)* **Neuromorphic hardware** for low-power wearables. Be concrete: "applied to this problem, this would allow…". |
| 8 | Conclusion | 100 | — | What you built, what you found, one honest limitation |
| 9 | References | — | — | ~15–25 references, mixed academic + commercial. **Not counted in word limit.** |

Academic vs commercial sources — the brief explicitly asks for both. Mix peer-reviewed papers with sources like Anthropic/OpenAI blogs, NVIDIA neurotech reports, Neurable/Emotiv product whitepapers, Gartner AI trend reports.

---

## 8. Key references to cite (starter set)

**Dataset + baselines:**
- Hinss, M.F. et al. (2022). *Open multi-session and multi-task EEG cognitive dataset for passive brain–computer interface applications.* Scientific Data.
- Roy, R.N. et al. (2016). ERP-based workload classification during MATB.

**Models:**
- Lawhern, V.J. et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based BCIs.* J. Neural Eng.
- Schirrmeister, R.T. et al. (2017). *Deep learning with convolutional neural networks for EEG decoding and visualization.* Hum. Brain Mapp.
- Song, Y. et al. (2022). *EEG Conformer.* IEEE TNSRE.

**Neuroscience of workload:**
- Bowers, M. et al. (2014); Ke, Y. et al. (2021) — frontal theta / parietal alpha as MW correlates
- Wickens, C.D. (2008) — Multiple Resource Theory

**Traditional ML baseline:**
- Congedo, M. et al. (2013). *Riemannian geometry for EEG-based BCI.*

**Trends (LO3):**
- Kostas, D. et al. (2021). *BENDR: Transformers for EEG pretraining.*
- Jiang, W. et al. (2024). *LaBraM: Large Brain Model.* ICLR 2024.
- Strubell, E. et al. (2019). *Energy and policy considerations for deep learning in NLP.* — or a newer equivalent for environmental impact

**Ethics / regulation:**
- EU AI Act (2024) — biometric categorisation provisions
- Ienca, M. & Andorno, R. (2017). *Towards new human rights in the age of neuroscience and neurotechnology.*

---

## 9. Environment + reproducibility

`requirements.txt`:
```
mne>=1.6
mne-bids>=0.14
numpy, pandas, scipy, matplotlib, seaborn
scikit-learn>=1.3
pyriemann>=0.5
torch>=2.1
captum>=0.7     # for Integrated Gradients
tqdm, pyyaml
```

- Fix seeds in numpy, torch, random, and `torch.use_deterministic_algorithms(True)` where possible.
- Log library versions in the first output cell.
- A Google Colab Pro or a single 8 GB GPU is enough. EEGNet trains in minutes per fold.
- Storage: download only the subject zips you need; unzip on a fast SSD.

---

## 10. Realistic week-by-week schedule

Assume today is ~4 weeks out from the 27 April deadline.

| Week | Focus | Output |
|---|---|---|
| **1 (this week)** | Get problem approved; download 3–5 subject zips; read BIDS structure; get MNE reading one subject end-to-end | Running data-load cell, class distribution plot |
| **2** | Preprocessing pipeline + traditional ML baselines on all subjects (LOSO); EDA figures | Baseline numbers in a results table |
| **3** | EEGNet + one other DL model trained; LOSO evaluation; interpretability figure | Final results table, confusion matrices, saliency figure |
| **4 (final week)** | Write reflection; polish notebook; proofread; submit with 24 h buffer | Submission |

Do **not** leave the reflection to the last 48 hours — it is where the marks live. Write section 2 (LO1) and section 7 (LO3) *first* since they depend on literature rather than your final numbers.

---

## 11. Pitfalls that specifically cap marks (from the rubric)

- **Random split leakage** → inflated numbers → examiner spots it → trust lost. Use LOSO.
- **No interpretability** → capped around 60 on LO2. Add saliency maps.
- **Only one model** → weak comparison → capped around 60 on LO1. Run at least 2 DL + 2 ML.
- **Generic "AI is transforming the world" future-trends section** → weak LO3. Tie each trend to *this* dataset and *this* model.
- **No ethics / environmental discussion** → capped on LO2. Cover individual + org + environment + society.
- **Transport framing** → direct breach of the brief. Frame as operator cognitive monitoring, not driving/aviation.
- **Missing references** → every rubric band below 60 explicitly lists "lack or insufficient references."
- **Word count violations >10%** → 10% mark penalty; >30% → capped at pass.

---

## 12. What to submit

1. `WM9B7_<your-ID>_notebook.ipynb` — self-contained, runnable top-to-bottom, with README cell, all outputs present (do not clear outputs before submitting).
2. `WM9B7_<your-ID>_reflection.pdf` — 2,800-word (max) critical reflection.
3. (Optional) `requirements.txt` and a short `README.md` alongside the notebook.

Upload both to Tabula before 12:00 on 27 April 2026.
