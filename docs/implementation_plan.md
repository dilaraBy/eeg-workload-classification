# EEG Workload Classification — Implementation Plan

**Current best**: EEGNet acc=0.529, macro F1=0.491  
**Target**: acc ≥ 0.65, macro F1 ≥ 0.60  
**Protocol**: Cross-session (train S1+S2, test S3), 29 subjects, 3-class MATB  
**Last updated**: 2026-04-23

---

## Baseline (already implemented)

| Component | Status | Details |
|---|---|---|
| Preprocessing | ✅ | Notch 50Hz, bandpass 1–40Hz, CAR, resample 250Hz, 6s windows 50% overlap |
| Bad channel handling | ✅ | Robust z-score detection + interpolation |
| Euclidean Alignment | ✅ | Applied globally to train and test sets |
| Z-score normalisation | ✅ | Train stats only, applied to test |
| SVM + band-power | ✅ | θ/α/β, 62ch × 3 bands = 186 features, RBF kernel, C=2 |
| EEGNet | ✅ | F1=8, D=2, dropout=0.5, lr=1e-3 |
| DeepConvNet | ✅ | 3 conv blocks, dropout=0.4, lr=3e-4 |
| FocalLoss + LabelSmoothing | ✅ | γ=2.0, smoothing=0.1, class-weighted |
| Per-subject breakdown | ✅ | Section 13 |
| Grad-CAM saliency | ✅ | Section 14, DeepConvNet block3 ELU |

---

## Tier 1 — Quick wins (implement first)

### T1-A: Fix DeepConvNet focal gamma → 1.0
**File**: Section 9 training cell  
**Expected gain**: +3–5% Low recall for DeepConvNet, better balance  
**Effort**: Trivial (1 line)

γ=2 was too aggressive for DeepConvNet — Low recall dropped to 0.500. EEGNet handled γ=2 well. Run DeepConvNet with γ=1.0 as a separate condition.

```python
# In training cell, pass gamma override per model:
DEEPCONVNET_FOCAL_GAMMA = 1.0
EEGNET_FOCAL_GAMMA = 2.0

# Update FocalLoss instantiation in _build_criterion() to accept gamma argument
# OR just set FOCAL_GAMMA=1.0 before training DeepConvNet and reset after
```

**Implementation steps**:
1. Add `model_gamma` parameter to `_build_criterion(gamma)` 
2. Pass `DEEPCONVNET_FOCAL_GAMMA` when calling `train_and_eval` for DeepConvNet
3. Keep γ=2.0 for EEGNet
4. Compare DeepConvNet results before and after

---

### T1-B: Cosine annealing LR with linear warmup
**File**: Section 9 training cell — `train_and_eval` function  
**Expected gain**: +1–3% DL accuracy, more stable convergence  
**Effort**: Low (~10 lines)

Replace `ReduceLROnPlateau` with a linear warmup followed by cosine decay. Standard best practice for transformer and conv models. Particularly improves EEGNet which is LR-sensitive.

```python
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

WARMUP_EPOCHS = 5  # linear ramp from 0 to lr over first 5 epochs

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

# In train_and_eval, replace scheduler with:
scheduler = get_cosine_warmup_scheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)
# And call scheduler.step() every epoch (not on val_loss)
```

**Implementation steps**:
1. Import `math` and `LambdaLR`
2. Add `get_cosine_warmup_scheduler` function above `train_and_eval`
3. Replace `ReduceLROnPlateau` with cosine warmup scheduler
4. Change `scheduler.step(va_loss)` → `scheduler.step()` (no argument)
5. Add `WARMUP_EPOCHS = 5` to hyperparameter config block

---

### T1-C: On-the-fly data augmentation in DataLoader
**File**: Section 9 training cell  
**Expected gain**: +2–4%, reduces cross-session overfitting  
**Effort**: Low (~30 lines)

Apply three augmentations with configurable probability. Applied only during training, never during validation or test.

```python
class EEGAugmenter:
    """On-the-fly EEG augmentation for (batch, 1, n_ch, n_times) tensors."""
    
    def __init__(
        self,
        noise_std: float = 0.05,
        shift_max_samples: int = 50,
        channel_dropout_p: float = 0.10,
        apply_prob: float = 0.5,
    ):
        self.noise_std = noise_std
        self.shift_max = shift_max_samples
        self.ch_drop_p = channel_dropout_p
        self.apply_prob = apply_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_ch, n_times)
        if torch.rand(1).item() > self.apply_prob:
            return x
        x = x.clone()

        # 1. Gaussian noise
        x = x + self.noise_std * torch.randn_like(x)

        # 2. Random temporal circular shift
        shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)

        # 3. Random channel dropout
        if self.ch_drop_p > 0:
            n_ch = x.shape[2]
            drop_mask = torch.rand(n_ch) < self.ch_drop_p
            x[:, :, drop_mask, :] = 0.0

        return x

AUGMENTER = EEGAugmenter(noise_std=0.05, shift_max_samples=50, channel_dropout_p=0.10)

# In the training loop, add augmentation before forward pass:
# with torch.amp.autocast(...):
#     xb_aug = AUGMENTER(xb)
#     logits = model(xb_aug)
```

**Implementation steps**:
1. Add `EEGAugmenter` class to Section 9 cell
2. Instantiate `AUGMENTER` with config flags
3. In the training loop body, apply `xb_aug = AUGMENTER(xb)` before `model(xb_aug)`
4. Validation/test forward passes keep `xb` unchanged
5. Add `AUG_ENABLED = True` flag for easy toggle

---

### T1-D: Riemannian tangent space features + SVM
**File**: New Section 7b cell (after Section 7, before Section 8)  
**Expected gain**: +8–15% accuracy over current SVM  
**Effort**: Low (requires `pip install pyriemann`)

This is the single biggest expected gain. Riemannian geometry on covariance matrices captures spatial correlations between channels that scalar band-power discards. It is the current SOTA baseline for cross-session EEG and what the COG-BCI paper benchmarks against.

```python
# pip install pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Covariance estimation
cov_estimator = Covariances(estimator="lwf")  # Ledoit-Wolf shrinkage
X_train_cov = cov_estimator.fit_transform(X_train_model)   # (n_epochs, n_ch, n_ch)
X_test_cov  = cov_estimator.transform(X_test_model)

# Project to tangent space (fit reference point on train only)
ts = TangentSpace(metric="riemann")
X_train_ts = ts.fit_transform(X_train_cov)   # (n_epochs, n_ch*(n_ch+1)/2)
X_test_ts  = ts.transform(X_test_cov)

# SVM on tangent space features
RIEM_SVM = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
])
RIEM_SVM.fit(X_train_ts, y_train)
y_pred_riem = RIEM_SVM.predict(X_test_ts)

riem_acc = accuracy_score(y_test, y_pred_riem)
riem_f1  = f1_score(y_test, y_pred_riem, average="macro")
RIEM_RESULTS = {"model": "Riemannian SVM", "accuracy": riem_acc, "macro_f1": riem_f1}
print(RIEM_RESULTS)
```

**Also try MDM (Minimum Distance to Mean)**:
```python
from pyriemann.classification import MDM

MDM_CLF = MDM(metric="riemann")
MDM_CLF.fit(X_train_cov, y_train)
y_pred_mdm = MDM_CLF.predict(X_test_cov)
```

**Implementation steps**:
1. Add `pip install pyriemann` to requirements.txt
2. Insert new markdown cell: `## 7b. Riemannian Geometry Features`
3. Insert code cell with covariance estimation + tangent space + SVM
4. Also run MDM for comparison
5. Add both to RESULTS_DF in Section 11

**Notes**:
- Use `estimator="lwf"` (Ledoit-Wolf) not `"oas"` — more stable for n_ch=62
- `X_train_model` is already EA-aligned, which is the correct input for Riemannian
- The tangent space dimension will be 62×63/2 = 1953 features — StandardScaler is essential

---

### T1-E: Soft-voting ensemble
**File**: New cell after Section 12 error analysis  
**Expected gain**: +2–5%  
**Effort**: Low (~20 lines)

Average probability outputs from best models. Works because SVM has better Low/High separation while DeepConvNet has better Medium recall — their errors are partially complementary.

```python
from sklearn.calibration import CalibratedClassifierCV

# Re-train SVM with probability=True (or use CalibratedClassifierCV)
SVM_PROB = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", CalibratedClassifierCV(
        SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"),
        cv=3, method="sigmoid"
    )),
])
SVM_PROB.fit(X_train_feat, y_train)
svm_proba = SVM_PROB.predict_proba(X_test_feat)  # (n_test, 3)

# Get softmax probabilities from EEGNet and DeepConvNet
def get_model_proba(model, loader, device):
    model.eval()
    all_proba = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
    return np.concatenate(all_proba, axis=0)

eegnet_proba     = get_model_proba(EEGNET_MODEL, test_loader, device)
deepconvnet_proba = get_model_proba(DEEPCONVNET_MODEL, test_loader, device)

# Weighted soft vote: weight by macro F1 on validation set
w_svm  = EEGNET_RESULTS["macro_f1"]  # use proportional weights
w_eeg  = EEGNET_RESULTS["macro_f1"]
w_deep = DEEPCONVNET_RESULTS["macro_f1"]
total  = w_svm + w_eeg + w_deep

ensemble_proba = (
    (w_svm  / total) * svm_proba +
    (w_eeg  / total) * eegnet_proba +
    (w_deep / total) * deepconvnet_proba
)
y_pred_ensemble = ensemble_proba.argmax(axis=1)

ENSEMBLE_RESULTS = {
    "model": "Soft-Vote Ensemble",
    "accuracy": float(accuracy_score(y_test, y_pred_ensemble)),
    "macro_f1": float(f1_score(y_test, y_pred_ensemble, average="macro")),
}
print(ENSEMBLE_RESULTS)
```

---

## Tier 2 — Moderate effort, significant gain

### T2-A: Expand frequency bands (delta + gamma)
**File**: Section 7 band-power extraction cell  
**Expected gain**: +2–4% for SVM and Riemannian  
**Effort**: Trivial (add to BANDS dict)

```python
BANDS = {
    "delta": (0.5, 4.0),   # fatigue, deep processing
    "theta": (4.0, 8.0),   # working memory load
    "alpha": (8.0, 13.0),  # cognitive inhibition
    "beta":  (13.0, 30.0), # active processing
    "gamma": (30.0, 45.0), # high cognitive load marker
}
# 62 channels × 5 bands = 310 features
```

**Implementation steps**:
1. Update `BANDS` dict in Section 7
2. Re-run band-power extraction — all downstream SVM code updates automatically
3. Compare SVM performance with 3-band vs 5-band features

---

### T2-B: Per-session Euclidean Alignment
**File**: Section 6 EA cell  
**Expected gain**: +2–4% — more accurate covariance estimate per session  
**Effort**: Low (~15 lines)

Currently EA computes one reference covariance over ALL train windows pooled. A per-session EA computes a separate reference per session (S1, S2) and aligns each independently before pooling. The per-session covariance estimate is more accurate because it avoids mixing session-specific distributions.

```python
def euclidean_alignment_per_session(
    X: np.ndarray,
    session_labels: np.ndarray,
) -> np.ndarray:
    """Apply EA independently per session, then concatenate."""
    X_out = np.empty_like(X)
    for sess in np.unique(session_labels):
        mask = session_labels == sess
        X_out[mask] = euclidean_alignment(X[mask])
    return X_out

# Build session label array for train windows from TRAIN_PREPROC_LOG
train_session_labels = []
for _, row in TRAIN_PREPROC_LOG.iterrows():
    train_session_labels.extend([row["session"]] * int(row["kept_windows"]))
train_session_arr = np.array(train_session_labels)

X_train_ea_persess = euclidean_alignment_per_session(X_train, train_session_arr)
# Test set: each subject only has S3, so standard EA applies
X_test_ea_persess  = euclidean_alignment(X_test)
```

---

### T2-C: ShallowConvNet
**File**: Section 9 model definitions cell  
**Expected gain**: Comparable to or better than DeepConvNet on 6s windows  
**Effort**: Medium (~40 lines)

Single large temporal filter + spatial filter + square nonlinearity + mean-pool + log. Explicitly models band-power in the network. Fast to train. Reference: Schirrmeister et al. (2017).

```python
class ShallowConvNet(nn.Module):
    """
    Schirrmeister et al. (2017) ShallowConvNet.
    Temporal conv (1×25) → spatial conv (n_ch×1) → square → mean-pool → log → FC.
    """
    def __init__(self, n_classes=3, n_channels=62, n_times=1500,
                 n_filters=40, filter_len=25, pool_len=75, pool_stride=15,
                 dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            1, n_filters, kernel_size=(1, filter_len), bias=False
        )
        self.spatial_conv = nn.Conv2d(
            n_filters, n_filters, kernel_size=(n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters, momentum=0.1, eps=1e-5)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_len), stride=(1, pool_stride))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self._features(dummy)
            feat_dim = out.flatten(1).shape[1]
        self.fc = nn.Linear(feat_dim, n_classes)

    def _features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2               # square activation
        x = self.pool(x)
        x = torch.log(x.clamp(min=1e-6))   # log activation
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.drop(x.flatten(1))
        return self.fc(x)

# Instantiate and train:
# SHALLOW_MODEL = ShallowConvNet(n_classes=3, n_channels=n_channels, n_times=n_times)
# SHALLOW_RESULTS, ..., SHALLOW_MODEL = train_and_eval(
#     SHALLOW_MODEL, "ShallowConvNet", lr=1e-3, weight_decay=5e-4
# )
```

---

### T2-D: SE (Squeeze-and-Excitation) channel attention for DeepConvNet
**File**: Section 9 model definitions  
**Expected gain**: +1–3%, teaches the model which 62 channels matter for workload  
**Effort**: Medium (~25 lines + modify DeepConvNet)

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for per-channel recalibration."""
    def __init__(self, n_channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, max(n_channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(n_channels // reduction, 1), n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, H, W)
        scale = x.mean(dim=(2, 3))        # global avg pool → (batch, C)
        scale = self.fc(scale)            # (batch, C)
        return x * scale[:, :, None, None]

class DeepConvNetSE(nn.Module):
    """DeepConvNet with Squeeze-and-Excitation channel attention after each block."""
    def __init__(self, n_classes=3, n_channels=62, dropout=0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(16, 16, (n_channels, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
        )
        self.se1 = SEBlock(16)
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
        )
        self.se2 = SEBlock(32)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
        )
        self.se3 = SEBlock(64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.se1(self.block1(x))
        x = self.se2(self.block2(x))
        x = self.se3(self.block3(x))
        return self.fc(self.gap(x).flatten(1))
```

---

### T2-E: CORAL domain adaptation (feature-level)
**File**: New cell between Section 7 and Section 8  
**Expected gain**: +3–6% on test set  
**Effort**: Medium (~30 lines)

CORAL aligns the second-order statistics (covariance) of train and test feature distributions. Applied to band-power or tangent-space features before SVM. No labels from the test set are needed — only the test feature matrix.

```python
def coral_align(X_source: np.ndarray, X_target: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    """
    CORAL: align X_source covariance to match X_target covariance.
    Returns X_source_aligned with same shape as X_source.
    Sun & Saenko (2016) ECCV.
    """
    ns, d = X_source.shape
    C_s = np.cov(X_source, rowvar=False) + eps * np.eye(d)
    C_t = np.cov(X_target, rowvar=False) + eps * np.eye(d)

    # Whitening: X_s @ C_s^{-1/2}
    vals_s, vecs_s = np.linalg.eigh(C_s)
    vals_s = np.maximum(vals_s, eps)
    W_s = vecs_s @ np.diag(1.0 / np.sqrt(vals_s)) @ vecs_s.T

    # Re-colour: X_s_white @ C_t^{1/2}
    vals_t, vecs_t = np.linalg.eigh(C_t)
    vals_t = np.maximum(vals_t, eps)
    W_t = vecs_t @ np.diag(np.sqrt(vals_t)) @ vecs_t.T

    A = W_s @ W_t   # combined transform (d, d)
    return (X_source @ A).astype(np.float32)

# Apply to band-power features:
X_train_feat_coral = coral_align(X_train_feat, X_test_feat)
# Then re-train SVM on X_train_feat_coral, predict on X_test_feat
```

---

## Tier 3 — Research-level (high effort, high reward)

### T3-A: EEGConformer (CNN + Transformer)
**File**: Section 9 model definitions  
**Expected gain**: +5–10% if it converges  
**Effort**: High (~100 lines + careful LR tuning)

Combines CNN patch embedding (like EEGNet) with multi-head self-attention for global temporal context. Best for 6s windows where long-range temporal dependencies matter. Critical: requires warmup LR, small weight decay (1e-5), and dropout ≥ 0.3 in the attention layers.

```python
class EEGConformer(nn.Module):
    """
    Song et al. (2023) EEGConformer.
    CNN front-end (patch embedding) + Transformer encoder + FC.
    """
    def __init__(self, n_classes=3, n_channels=62, n_times=1500,
                 patch_size=40, n_filters=40, n_heads=8,
                 n_layers=6, d_model=40, dropout=0.3):
        super().__init__()
        # CNN patch embedding (same as EEGNet front-end)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, patch_size), padding=(0, patch_size//2), bias=False),
            nn.Conv2d(n_filters, n_filters, (n_channels, 1), groups=n_filters, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, 1, n_ch, n_times)
        x = self.patch_embed(x)               # (B, F, 1, T')
        x = x.squeeze(2).transpose(1, 2)      # (B, T', F) — sequence for transformer
        x = self.transformer(x)               # (B, T', F)
        x = self.norm(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, F)
        return self.fc(self.drop(x))

# Training notes:
# - Use lr=3e-4 with WARMUP_EPOCHS=10 (more warmup than CNN models)
# - weight_decay=1e-5 (lower than CNN models)
# - MAX_EPOCHS=100, PATIENCE=15 (needs more epochs to converge)
```

---

### T3-B: DANN — Domain Adversarial Neural Network
**File**: Section 9 model definitions + training loop  
**Expected gain**: +4–8% cross-session  
**Effort**: High (requires custom training loop)

Adds a domain classifier branch to EEGNet that tries to predict which session (S1/S2 vs S3) a window came from. The feature extractor is trained adversarially to fool this branch via a gradient reversal layer — forcing it to learn session-invariant workload features.

```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class EEGNetDANN(nn.Module):
    def __init__(self, base_eegnet: EEGNet, feat_dim: int):
        super().__init__()
        self.feature_extractor = base_eegnet   # reuse EEGNet feature layers
        self.label_classifier  = nn.Linear(feat_dim, 3)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(),
            nn.Linear(32, 2),  # 2 domains: train sessions vs test session
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor._forward_features(x).flatten(1)
        label_output  = self.label_classifier(features)
        reversed_feat = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_feat)
        return label_output, domain_output

# Training loop modification:
# total_loss = label_loss + lambda_domain * domain_loss
# alpha increases from 0 → 1 over training (standard DANN schedule)
```

**Training requires**: session labels for each window (buildable from TRAIN_PREPROC_LOG + test label=1 for all S3 windows).

---

### T3-C: ICA artifact removal
**File**: Section 5 preprocessing cell (optional branch)  
**Expected gain**: +1–4% mainly for noisy subjects (sub-02 had 27% window retention)  
**Effort**: Medium-high (slow, ~5min per subject)

The `core+ICA` variant is already stubbed in the codebase. Implementation:

```python
def apply_ica(raw: mne.io.BaseRaw, n_components: int = 30,
              method: str = "fastica") -> mne.io.BaseRaw:
    """Run ICA and auto-reject ocular/muscle components."""
    ica = mne.preprocessing.ICA(
        n_components=n_components, method=method,
        max_iter="auto", random_state=SEED,
    )
    ica.fit(raw, verbose="ERROR")

    # Auto-detect eye and muscle components
    eog_indices, _ = ica.find_bads_eog(raw, threshold=3.0, verbose="ERROR")
    muscle_indices, _ = ica.find_bads_muscle(raw, threshold=0.5, verbose="ERROR")
    ica.exclude = list(set(eog_indices + muscle_indices))[:6]  # cap at 6 components

    ica.apply(raw, verbose="ERROR")
    return raw

# Add PREPROC_VARIANT = "core+ICA" option:
# if PREPROC_VARIANT == "core+ICA":
#     raw_clean = apply_ica(raw_clean)
```

**Note**: Only compare ICA vs core on the same model (DeepConvNet) to isolate preprocessing effect.

---

## Hyperparameter search

### Grid search for SVM (band-power and Riemannian)
```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Run on train set only
svm_grid = {
    "svm__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "svm__gamma": ["scale", "auto"],
    "svm__kernel": ["rbf"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(SVM_PIPELINE, svm_grid, cv=cv, scoring="f1_macro",
                            n_jobs=-1, verbose=1)
grid_search.fit(X_train_feat, y_train)
print("Best SVM params:", grid_search.best_params_)
print("Best CV macro F1:", grid_search.best_score_)
```

### Focal gamma sweep for DL models
```python
GAMMA_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5]
gamma_results = []
for gamma in GAMMA_SWEEP:
    model = DeepConvNet(n_classes=3, n_channels=n_channels, dropout=0.4)
    # Temporarily override FOCAL_GAMMA
    original = FOCAL_GAMMA
    globals()["FOCAL_GAMMA"] = gamma
    res, _, _, _, _, _ = train_and_eval(model, f"DCN_gamma{gamma}", LR_DEEPCONVNET)
    globals()["FOCAL_GAMMA"] = original
    gamma_results.append({"gamma": gamma, **res})
display(pd.DataFrame(gamma_results).round(4))
```

---

## Implementation order and timeline

```
Day 1 (fastest wins):
  T1-A  Fix DeepConvNet gamma → 1.0             ~10 min
  T1-B  Cosine LR warmup                        ~20 min
  T1-C  Data augmentation                       ~30 min
  T1-D  Riemannian tangent space + SVM          ~30 min
  T1-E  Soft-voting ensemble                    ~20 min
  → Expected: acc ≈ 0.57–0.62

Day 2 (architecture + features):
  T2-A  Expand to 5 frequency bands             ~10 min
  T2-B  Per-session EA                          ~20 min
  T2-C  ShallowConvNet                          ~40 min
  T2-D  SE attention in DeepConvNet             ~30 min
  T2-E  CORAL domain adaptation                 ~30 min
  → Expected: acc ≈ 0.62–0.67

Day 3 (research-level):
  T3-A  EEGConformer                            ~2–3 hr
  T3-B  DANN (optional, high risk/reward)       ~3–4 hr
  T3-C  ICA (optional)                          ~1–2 hr
  → Expected: acc ≈ 0.65–0.72
```

---

## Progress tracker

| ID | Method | Status | Acc | Macro F1 | Medium Recall | Notes |
|---|---|---|---|---|---|---|
| baseline | SVM + band-power | ✅ done | 0.5134 | 0.4770 | 0.357 | 3-band, γ=2 |
| baseline | EEGNet | ✅ done | 0.5292 | 0.4906 | 0.241 | γ=2 |
| baseline | DeepConvNet | ✅ done | 0.4639 | 0.4542 | 0.404 | γ=2 too high |
| T1-A | DeepConvNet γ=1.0 | ⬜ todo | — | — | — | |
| T1-B | Cosine LR warmup | ⬜ todo | — | — | — | |
| T1-C | Augmentation | ⬜ todo | — | — | — | |
| T1-D | Riemannian SVM | ⬜ todo | — | — | — | |
| T1-D | MDM classifier | ⬜ todo | — | — | — | |
| T1-E | Ensemble | ⬜ todo | — | — | — | |
| T2-A | 5-band features | ⬜ todo | — | — | — | |
| T2-B | Per-session EA | ⬜ todo | — | — | — | |
| T2-C | ShallowConvNet | ⬜ todo | — | — | — | |
| T2-D | SE-DeepConvNet | ⬜ todo | — | — | — | |
| T2-E | CORAL | ⬜ todo | — | — | — | |
| T3-A | EEGConformer | ⬜ todo | — | — | — | |
| T3-B | DANN | ⬜ todo | — | — | — | |
| T3-C | ICA | ⬜ todo | — | — | — | |

---

## Key references

- Schirrmeister et al. (2017) — DeepConvNet + ShallowConvNet. *Human Brain Mapping*
- Lawhern et al. (2018) — EEGNet. *Journal of Neural Engineering*
- Barachant et al. (2012) — MDM classifier. *IEEE Trans Biomed Eng*
- Song et al. (2023) — EEGConformer. *IEEE Trans Neural Syst Rehabil Eng*
- Ganin et al. (2016) — DANN. *JMLR*
- Sun & Saenko (2016) — CORAL. *ECCV*
- He et al. (2016) — SE Networks. *CVPR*
- Lin et al. (2017) — Focal Loss. *ICCV*
- Hinss et al. (2023) — COG-BCI dataset. *Scientific Data*
