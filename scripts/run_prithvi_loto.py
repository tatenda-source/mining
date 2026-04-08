#!/usr/bin/env python3
"""
Automated Prithvi-EO-2.0 LOTO CV Training
==========================================
Runs entirely locally on Apple Silicon MPS GPU.
No Colab, no uploads, no manual steps.

Usage:
    python scripts/run_prithvi_loto.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter, defaultdict
import time

# ── Device setup ──
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Device: Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Device: CUDA GPU ({torch.cuda.get_device_name()})")
else:
    DEVICE = torch.device("cpu")
    print(f"Device: CPU (will be slow)")

# ── Paths ──
PROJECT = Path(__file__).resolve().parent.parent
CHIPS_DIR = PROJECT / "data" / "processed" / "chips"
LABELS_DIR = PROJECT / "data" / "processed" / "chip_labels"
META_PATH = PROJECT / "data" / "processed" / "chip_meta.json"
OUTPUT_DIR = PROJECT / "data" / "outputs"

# ── HLS normalization stats from Prithvi pre-training ──
# Our chips are reflectance 0-1, HLS stats are 0-10000 scale
PRITHVI_MEANS = np.array([775.2290, 1080.9920, 1228.5855, 2497.2180, 2204.2412, 1610.8815]) / 10000.0
PRITHVI_STDS  = np.array([1281.5260, 1270.4814, 1399.4836, 1368.3446, 1291.6764, 1154.5053]) / 10000.0


class MineralChipDataset(Dataset):
    """Dataset of 6-band Sentinel-2 chips with binary deposit masks."""
    def __init__(self, meta, chips_dir, labels_dir):
        self.meta = meta
        self.chips_dir = Path(chips_dir)
        self.labels_dir = Path(labels_dir)
        self.means = torch.tensor(PRITHVI_MEANS, dtype=torch.float32).view(6, 1, 1)
        self.stds = torch.tensor(PRITHVI_STDS, dtype=torch.float32).view(6, 1, 1)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        rec = self.meta[idx]
        chip_path = self.chips_dir / f"{rec['chip_id']}.tif"
        label_path = self.labels_dir / f"{rec['chip_id']}.tif"

        import rasterio
        with rasterio.open(chip_path) as src:
            chip = src.read().astype(np.float32)  # (6, 224, 224)
        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.float32)  # (224, 224)

        chip = torch.from_numpy(chip)
        chip = (chip - self.means) / self.stds  # HLS normalization
        mask = torch.from_numpy(mask)

        # Binary chip-level label: does this chip contain any deposit pixels?
        has_deposit = 1.0 if mask.sum() > 0 else 0.0

        return chip, mask, has_deposit


class ViTMineralClassifier(nn.Module):
    """
    Vision Transformer for mineral prospectivity.
    Uses timm's ViT backbone (pre-trained on ImageNet, adapted to 6 bands).
    This is a simpler but immediately runnable alternative to full Prithvi,
    which requires the NASA package that may have compatibility issues.

    Architecture:
    - ViT-Small backbone (22M params vs Prithvi's 300M -- faster, still spatial)
    - 6-channel input adapter (projects 6 bands -> 3 for ViT, or custom patch embed)
    - Binary classification head (deposit vs no-deposit per chip)
    - Also outputs a coarse spatial attention map for interpretability
    """
    def __init__(self):
        super().__init__()
        import timm

        # Input adapter: project 6 Sentinel-2 bands to 3 channels
        self.band_adapter = nn.Sequential(
            nn.Conv2d(6, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )

        # ViT backbone (pre-trained on ImageNet)
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=0,  # remove classifier head
        )

        # Classification head
        embed_dim = self.backbone.embed_dim  # 384 for vit_small
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Spatial attention head (for interpretability)
        # Uses patch tokens to produce a coarse 14x14 attention map
        self.spatial_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        B = x.shape[0]

        # Adapt 6 bands -> 3 channels
        x = self.band_adapter(x)

        # Get ViT features
        features = self.backbone.forward_features(x)  # (B, num_tokens, embed_dim)

        # CLS token for classification
        cls_token = features[:, 0]  # (B, embed_dim)
        logit = self.classifier(cls_token).squeeze(-1)  # (B,)

        # Patch tokens for spatial attention
        patch_tokens = features[:, 1:]  # (B, 196, embed_dim)
        attn_map = self.spatial_head(patch_tokens).squeeze(-1)  # (B, 196)
        attn_map = attn_map.view(B, 14, 14)  # (B, 14, 14)

        return logit, attn_map


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for chip, mask, label in loader:
        chip = chip.to(device)
        label = label.float().to(device)

        logit, _ = model(chip)
        loss = criterion(logit, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * chip.size(0)
        pred = (torch.sigmoid(logit) > 0.5).float()
        correct += (pred == label).sum().item()
        total += chip.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for chip, mask, label in loader:
            chip = chip.to(device)
            logit, _ = model(chip)
            prob = torch.sigmoid(logit)
            all_labels.extend(label.numpy().tolist())
            all_probs.extend(prob.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def compute_pr_auc(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return float('nan')
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return auc(rec, prec)


def main():
    print("=" * 70)
    print("AUTOMATED PRITHVI-STYLE LOTO CV")
    print("ViT-Small + 6-band adapter | Apple Silicon MPS")
    print("=" * 70)

    # Load metadata
    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"\nTotal chips: {len(meta)}")
    tile_label = Counter(f"{m['tile']}_{m['label']}" for m in meta)
    for k in sorted(tile_label):
        print(f"  {k}: {tile_label[k]}")

    tiles = sorted(set(m['tile'] for m in meta))
    print(f"\nTiles: {tiles}")
    print(f"Device: {DEVICE}")

    # ── LOTO Cross-Validation ──
    results = {}
    all_global_true = []
    all_global_prob = []

    for fold_idx, held_out in enumerate(tiles):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx+1}/{len(tiles)}: Hold out {held_out}")
        print(f"{'=' * 70}")

        train_meta = [m for m in meta if m['tile'] != held_out]
        test_meta = [m for m in meta if m['tile'] == held_out]

        train_pos = sum(1 for m in train_meta if m['label'] == 'positive')
        train_neg = sum(1 for m in train_meta if m['label'] == 'negative')
        test_pos = sum(1 for m in test_meta if m['label'] == 'positive')
        test_neg = sum(1 for m in test_meta if m['label'] == 'negative')

        print(f"  Train: {train_pos}+ / {train_neg}- = {len(train_meta)}")
        print(f"  Test:  {test_pos}+ / {test_neg}- = {len(test_meta)}")

        if test_pos == 0:
            print(f"  SKIP: no positive chips in test")
            continue

        train_ds = MineralChipDataset(train_meta, CHIPS_DIR, LABELS_DIR)
        test_ds = MineralChipDataset(test_meta, CHIPS_DIR, LABELS_DIR)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

        # Initialize model
        model = ViTMineralClassifier().to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold_idx == 0:
            print(f"  Model: {total_params/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")

        # Class imbalance weighting
        pos_weight = torch.tensor([train_neg / max(train_pos, 1)]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Stage 1: Freeze backbone, train adapter + head (5 epochs)
        for p in model.backbone.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=0.01
        )

        print(f"\n  Stage 1: Training adapter + head (backbone frozen)...")
        t0 = time.time()
        for epoch in range(5):
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            print(f"    Epoch {epoch+1}/5: loss={loss:.4f} acc={acc:.1%}")

        # Stage 2: Unfreeze backbone, fine-tune end-to-end (10 epochs, lower LR)
        for p in model.backbone.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        print(f"\n  Stage 2: Fine-tuning end-to-end...")
        for epoch in range(10):
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/10: loss={loss:.4f} acc={acc:.1%} lr={lr:.2e}")

        elapsed = time.time() - t0
        print(f"\n  Training time: {elapsed:.0f}s")

        # Evaluate
        y_true, y_prob = evaluate(model, test_loader, DEVICE)
        pr_auc = compute_pr_auc(y_true, y_prob)
        baseline = np.mean(y_true)
        lift = pr_auc / baseline if baseline > 0 else 0

        print(f"\n  >> PR-AUC: {pr_auc:.3f} (baseline: {baseline:.3f}, lift: {lift:.1f}x)")

        results[held_out] = {
            'pr_auc': float(pr_auc),
            'baseline': float(baseline),
            'lift': float(lift),
            'test_pos': test_pos,
            'test_neg': test_neg,
            'train_pos': train_pos,
            'train_neg': train_neg,
            'time_s': elapsed,
        }

        all_global_true.extend(y_true.tolist())
        all_global_prob.extend(y_prob.tolist())

    # ── Final Summary ──
    print(f"\n{'=' * 70}")
    print("LOTO CV RESULTS SUMMARY")
    print(f"{'=' * 70}")

    for tile, r in results.items():
        status = "PASS" if r['pr_auc'] >= 0.45 else "FAIL"
        print(f"  {tile}: PR-AUC={r['pr_auc']:.3f} (baseline={r['baseline']:.3f}, "
              f"lift={r['lift']:.1f}x) [{status}]")

    valid = [r for r in results.values() if np.isfinite(r['pr_auc'])]
    if valid:
        mean_auc = np.mean([r['pr_auc'] for r in valid])
        weighted_auc = sum(r['pr_auc'] * r['test_pos'] for r in valid) / sum(r['test_pos'] for r in valid)
        total_time = sum(r['time_s'] for r in valid)

        print(f"\n  Mean LOTO PR-AUC:     {mean_auc:.3f}")
        print(f"  Weighted LOTO PR-AUC: {weighted_auc:.3f}")
        print(f"  Total training time:  {total_time:.0f}s ({total_time/60:.1f} min)")

        # Pooled PR-AUC
        if sum(all_global_true) > 0:
            pooled = compute_pr_auc(all_global_true, all_global_prob)
            pooled_bl = np.mean(all_global_true)
            print(f"  Pooled PR-AUC:        {pooled:.3f} (baseline: {pooled_bl:.3f})")

        print(f"\n  Target: >= 0.45")
        print(f"  Verdict: {'PASS -- CROSS-TILE SIGNAL DETECTED' if mean_auc >= 0.45 else 'FAIL -- signal does not transfer'}")

        if mean_auc >= 0.45:
            print(f"\n  *** THE NUMBER IS DEFENSIBLE. PITCH DECK READY. ***")
        elif mean_auc >= 0.20:
            print(f"\n  Partial signal detected. Foundation model helps but needs more data/epochs.")
        else:
            print(f"\n  No improvement over logistic regression. Consider:")
            print(f"    - More training epochs")
            print(f"    - Larger ViT (vit_base_patch16_224)")
            print(f"    - Multi-temporal chips (seasonal variation)")
            print(f"    - Actual Prithvi weights (pre-trained on satellite data, not ImageNet)")

    # Save results
    output = {
        'model': 'ViT-Small + 6-band adapter',
        'cv': 'LOTO (tile-based)',
        'device': str(DEVICE),
        'folds': results,
        'mean_pr_auc': float(mean_auc) if valid else None,
    }
    out_path = OUTPUT_DIR / "prithvi_loto_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
