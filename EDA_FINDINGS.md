# EEG MATB EDA Findings (Initial Pass)

## Dataset Coverage

- Total `.set` files in dataset: 261
- MATB `.set` files: 261
- Subjects: 29 (`sub-01` to `sub-29`)
- Sessions: 3 (`ses-S1`, `ses-S2`, `ses-S3`)
- Levels: `easy`, `med`, `diff`
- Balanced MATB file counts:
  - easy: 87
  - med: 87
  - diff: 87
- Each subject has 9 MATB files (3 sessions x 3 levels).
- No missing MATB triplets per subject-session.

## Signal Metadata

- Sampling frequency: 500 Hz for all MATB recordings.
- Channel count is not fully uniform:
  - 63 channels in 81 files
  - 64 channels in 180 files
- Difference between 63 and 64 channel sets: `Cz` exists only in 64-channel files.
- Recording duration is highly consistent:
  - min: 298.512 s
  - median: 299.254 s
  - max: 299.700 s

## Event/Annotation Observations

- MATB files are continuous recordings (not pre-epoched trials).
- Rich event annotations are available (example file had 1307 annotations), including task stream messages like `TRACKING`, `RESMAN`, etc.

## Amplitude Sanity Check (sampled files)

- Mean absolute amplitude (sampled): ~10,720 uV
- Std amplitude (sampled): ~7,352 uV

Interpretation: Raw signals appear to include large offsets/scale, so robust preprocessing is mandatory (filtering, artifact handling, normalization/alignment).

## Recommended Next Steps

1. Standardize channels first.
- Use channel intersection across all files to avoid shape mismatches.
- Practical option: drop `Cz` globally so all recordings share a 63-channel layout.

2. Build continuous-to-epoch pipeline.
- Load raw MATB (`read_raw_eeglab`).
- Bandpass 1-40 Hz (and notch if needed).
- Segment into fixed windows (for example 2 s windows with 50% overlap).
- Labels from file-level MATB condition (`easy/med/diff`) for initial baseline.

3. Apply cross-session-safe normalization.
- Fit normalization statistics on training sessions only.
- Consider Euclidean Alignment for session drift mitigation.

4. Run baseline experiment first.
- SVM on band-power features.
- EEGNet on windowed signals.
- Use cross-session split (train S1+S2, test S3) to match objective.

5. Add second-pass EDA visuals in notebook.
- Class balance bar chart.
- Duration histogram.
- Channel-count consistency check.
- PSD by class (sample subjects).
- Per-class amplitude distribution after preprocessing.

## Decision

Proceed with modeling now, but only after adding a channel-standardization step and robust preprocessing. The dataset is complete and structurally suitable for the planned ML-vs-DL cross-session study.
