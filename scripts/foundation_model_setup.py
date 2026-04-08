#!/usr/bin/env python3
"""
foundation_model_setup.py
=========================
Boilerplate for fine-tuning NASA/IBM Prithvi-EO-2.0 (and Clay) foundation
models for mineral prospectivity mapping on the Great Dyke, Zimbabwe.

This script provides:
  1. Dependency checking and model weight downloading
  2. Sentinel-2 data preparation for the model's expected input format
  3. A custom PyTorch dataset for binary segmentation (deposit vs non-deposit)
  4. A complete fine-tuning loop using TerraTorch / PyTorch Lightning

Usage:
  # Check dependencies and download weights only
  python scripts/foundation_model_setup.py --check-only

  # Prepare data chips from existing Sentinel-2 stacks
  python scripts/foundation_model_setup.py --prepare-data

  # Run fine-tuning (requires GPU for practical speed; works on CPU for testing)
  python scripts/foundation_model_setup.py --train

Compute requirements:
  - Prithvi-EO-2.0-300M: ~4 GB VRAM for batch_size=4, 224x224 chips
  - Prithvi-EO-2.0-600M: ~8 GB VRAM (use Colab T4 or better)
  - CPU: Works but extremely slow (~10x). Fine for data prep, not training.
  - Apple Silicon MPS: Partially supported by PyTorch. May work with caveats.
  - Recommended: Google Colab (free T4 GPU) or Colab Pro (A100).

Author: GeoMine AI Project
Date: April 2026
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_S2 = DATA_DIR / "processed" / "sentinel2"
TRAINING_DIR = DATA_DIR / "training"
CHIPS_DIR = DATA_DIR / "processed" / "chips"        # output of chip generation
LABELS_DIR = DATA_DIR / "processed" / "chip_labels"  # binary mask chips

# Sentinel-2 tiles available in this project
S2_TILES = ["T35KRU", "T36KTC", "T36KTD", "T36KTE"]

# ---------------------------------------------------------------------------
# Prithvi-EO-2.0 configuration
# ---------------------------------------------------------------------------
# Prithvi expects 6 HLS bands in this order: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2
# These map to Sentinel-2 bands:
#   Blue      -> B02 (490 nm)
#   Green     -> B03 (560 nm)
#   Red       -> B04 (665 nm)
#   Narrow NIR -> B8A (865 nm)   ** NOT B08 (842 nm) **
#   SWIR1     -> B11 (1610 nm)
#   SWIR2     -> B12 (2190 nm)
#
# Our stacked GeoTIFFs have bands: [B02, B03, B04, B08, B8A, B11, B12]
# So we need band indices:            0    1    2   skip  4    5    6
PRITHVI_BAND_INDICES = [0, 1, 2, 4, 5, 6]  # indices into our 7-band stack
PRITHVI_BAND_NAMES = ["Blue", "Green", "Red", "Narrow_NIR", "SWIR1", "SWIR2"]

# HLS normalization statistics from Prithvi-EO-2.0 pre-training
# These are per-band mean and std for HLS surface reflectance (scaled 0-10000)
# Source: ibm-nasa-geospatial/Prithvi-EO-2.0-300M config on HuggingFace
PRITHVI_MEANS = [775.2290, 1080.9920, 1228.5855, 2497.2180, 2204.2412, 1610.8815]
PRITHVI_STDS  = [1281.5260, 1270.4814, 1399.4836, 1368.3446, 1291.6764, 1154.5053]

# Model identifiers on HuggingFace
PRITHVI_300M_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
PRITHVI_300M_TL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"  # with temporal/location
PRITHVI_600M_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"

# Clay Foundation Model identifier
CLAY_MODEL_ID = "made-with-clay/Clay"

# Chip size for training patches (must be divisible by model's patch size, typically 16)
CHIP_SIZE = 224  # 224x224 pixels -- standard for ViT-based models
CHIP_STRIDE = 112  # 50% overlap for data augmentation via overlapping chips


# ===========================================================================
# Section 1: Dependency Checking
# ===========================================================================

def check_dependencies() -> dict:
    """
    Check whether all required packages for foundation model fine-tuning
    are installed and report their versions.

    Returns dict mapping package names to (installed: bool, version: str|None).
    """
    packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "lightning": "lightning",
        "terratorch": "terratorch",
        "timm": "timm",
        "segmentation_models_pytorch": "segmentation_models_pytorch",
        "einops": "einops",
        "huggingface_hub": "huggingface_hub",
        "safetensors": "safetensors",
        "rasterio": "rasterio",
        "geopandas": "geopandas",
        "rioxarray": "rioxarray",
    }

    results = {}
    for display_name, import_name in packages.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            results[display_name] = (True, version)
        except ImportError:
            results[display_name] = (False, None)

    return results


def print_dependency_report(results: dict) -> bool:
    """Print a human-readable dependency report. Returns True if all OK."""
    print("\n" + "=" * 60)
    print("Foundation Model Dependency Check")
    print("=" * 60)

    all_ok = True
    for name, (installed, version) in results.items():
        status = f"OK  v{version}" if installed else "MISSING"
        marker = "[+]" if installed else "[X]"
        print(f"  {marker} {name:30s} {status}")
        if not installed:
            all_ok = False

    # Check for GPU availability
    print("\n--- Compute Backend ---")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  [+] CUDA GPU: {gpu_name} ({vram:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  [~] Apple Silicon MPS: available (partial PyTorch support)")
            print("      Note: MPS works for inference. Training may hit unsupported ops.")
            print("      Fallback: set PYTORCH_ENABLE_MPS_FALLBACK=1 for CPU fallback on errors.")
        else:
            print("  [-] No GPU detected. CPU-only mode.")
            print("      Training will be very slow. Consider Google Colab (free T4 GPU).")
    except ImportError:
        print("  [-] PyTorch not installed -- cannot check GPU.")

    print("=" * 60)

    if not all_ok:
        print("\nTo install missing packages:")
        print("  pip install -r requirements-foundation.txt")
        print("\nGDAL is required by terratorch. If missing:")
        print("  conda install -c conda-forge gdal")

    return all_ok


# ===========================================================================
# Section 2: Model Weight Download
# ===========================================================================

def download_prithvi_weights(model_id: str = PRITHVI_300M_TL_ID) -> Path:
    """
    Download Prithvi-EO-2.0 pre-trained weights from HuggingFace.

    We use the 300M-TL variant (with Temporal and Location embeddings)
    because it learns better spatial representations that help with
    cross-tile generalization -- exactly our Phase 1 failure mode.

    Args:
        model_id: HuggingFace model ID. Default is 300M-TL (recommended).

    Returns:
        Path to the local cache directory containing model files.
    """
    from huggingface_hub import snapshot_download

    print(f"\nDownloading model: {model_id}")
    print("This may take a few minutes on first run (~1.2 GB for 300M)...")

    local_dir = snapshot_download(
        repo_id=model_id,
        # Only download model weights and config, skip large training artifacts
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.py", "*.md"],
    )

    print(f"Model cached at: {local_dir}")
    return Path(local_dir)


# ===========================================================================
# Section 3: Data Preparation
# ===========================================================================

def prepare_sentinel2_for_prithvi(
    tile_path: Path,
    output_chips_dir: Path,
    output_labels_dir: Path,
    deposits_path: Path,
    chip_size: int = CHIP_SIZE,
    chip_stride: int = CHIP_STRIDE,
    deposit_buffer_m: float = 500.0,
) -> int:
    """
    Convert a 7-band Sentinel-2 stacked GeoTIFF into Prithvi-compatible chips.

    Steps:
    1. Read the 7-band stack (B02, B03, B04, B08, B8A, B11, B12)
    2. Select the 6 bands Prithvi expects (drop B08, keep B8A)
    3. Slice into chip_size x chip_size patches
    4. Create binary label masks (1 = within buffer of deposit, 0 = background)
    5. Save each chip and label as separate GeoTIFFs

    Args:
        tile_path: Path to stacked Sentinel-2 GeoTIFF.
        output_chips_dir: Where to save image chips.
        output_labels_dir: Where to save label masks.
        deposits_path: Path to deposits GeoJSON (points).
        chip_size: Patch size in pixels.
        chip_stride: Stride between patches.
        deposit_buffer_m: Buffer radius around deposits in meters.

    Returns:
        Number of chips created.
    """
    import rasterio
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box

    output_chips_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    tile_name = tile_path.stem.replace("_stacked", "")

    # --- Read the stacked GeoTIFF ---
    with rasterio.open(tile_path) as src:
        # Read all bands -> shape (n_bands, height, width)
        data = src.read()
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    print(f"  Tile {tile_name}: {data.shape[0]} bands, {data.shape[1]}x{data.shape[2]} pixels")

    # --- Select the 6 Prithvi bands from our 7-band stack ---
    # Our stack: [B02, B03, B04, B08, B8A, B11, B12] (indices 0-6)
    # Prithvi:   [B02, B03, B04, B8A, B11, B12]       (indices 0,1,2,4,5,6)
    prithvi_data = data[PRITHVI_BAND_INDICES]  # shape: (6, H, W)
    n_bands, height, width = prithvi_data.shape

    # --- Load deposit locations and create rasterized label ---
    deposits = gpd.read_file(deposits_path)
    if deposits.crs != crs:
        deposits = deposits.to_crs(crs)

    # Buffer deposit points by deposit_buffer_m
    deposits_buffered = deposits.copy()
    deposits_buffered["geometry"] = deposits.geometry.buffer(deposit_buffer_m)

    # Rasterize: 1 where deposit buffer overlaps, 0 elsewhere
    label_raster = rasterize(
        [(geom, 1) for geom in deposits_buffered.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    n_positive_pixels = int(label_raster.sum())
    print(f"  Positive pixels (within {deposit_buffer_m}m of deposits): {n_positive_pixels}")

    # --- Chip extraction ---
    chip_count = 0
    for y in range(0, height - chip_size + 1, chip_stride):
        for x in range(0, width - chip_size + 1, chip_stride):
            chip_img = prithvi_data[:, y:y + chip_size, x:x + chip_size]
            chip_lbl = label_raster[y:y + chip_size, x:x + chip_size]

            # Skip chips that are mostly nodata (zeros)
            valid_fraction = np.count_nonzero(chip_img[0]) / (chip_size * chip_size)
            if valid_fraction < 0.8:
                continue

            # Compute the geo-transform for this chip
            chip_transform = rasterio.transform.from_bounds(
                transform.c + x * transform.a,
                transform.f + (y + chip_size) * transform.e,
                transform.c + (x + chip_size) * transform.a,
                transform.f + y * transform.e,
                chip_size,
                chip_size,
            )

            # Save image chip (6-band GeoTIFF)
            chip_name = f"{tile_name}_chip_{y:05d}_{x:05d}"
            chip_profile = profile.copy()
            chip_profile.update(
                count=n_bands,
                height=chip_size,
                width=chip_size,
                transform=chip_transform,
                dtype="float32",
            )
            with rasterio.open(output_chips_dir / f"{chip_name}.tif", "w", **chip_profile) as dst:
                dst.write(chip_img.astype(np.float32))

            # Save label chip (single-band mask)
            lbl_profile = chip_profile.copy()
            lbl_profile.update(count=1, dtype="uint8")
            with rasterio.open(output_labels_dir / f"{chip_name}.tif", "w", **lbl_profile) as dst:
                dst.write(chip_lbl[np.newaxis, :, :])

            chip_count += 1

    print(f"  Created {chip_count} chips from tile {tile_name}")
    return chip_count


def prepare_all_tiles():
    """Prepare chips from all available Sentinel-2 tiles."""
    deposits_path = TRAINING_DIR / "deposits_expanded.geojson"

    if not deposits_path.exists():
        # Fall back to curated deposits
        deposits_path = TRAINING_DIR / "deposits_curated.geojson"
    if not deposits_path.exists():
        deposits_path = TRAINING_DIR / "deposits.geojson"
    if not deposits_path.exists():
        print("ERROR: No deposits GeoJSON found in data/training/")
        print("Expected one of: deposits_expanded.geojson, deposits_curated.geojson, deposits.geojson")
        sys.exit(1)

    print(f"\nUsing deposits from: {deposits_path}")

    total_chips = 0
    for tile in S2_TILES:
        tile_path = PROCESSED_S2 / f"{tile}_stacked.tif"
        if not tile_path.exists():
            print(f"  Skipping {tile} -- file not found")
            continue
        total_chips += prepare_sentinel2_for_prithvi(
            tile_path=tile_path,
            output_chips_dir=CHIPS_DIR,
            output_labels_dir=LABELS_DIR,
            deposits_path=deposits_path,
        )

    print(f"\nTotal chips created: {total_chips}")
    print(f"  Image chips: {CHIPS_DIR}")
    print(f"  Label masks: {LABELS_DIR}")


# ===========================================================================
# Section 4: Custom Dataset for Mineral Prospectivity
# ===========================================================================

def build_dataset_and_dataloader():
    """
    Build a PyTorch Dataset and DataLoader for the mineral prospectivity
    segmentation task, compatible with Prithvi-EO-2.0 input format.

    This uses torchgeo's generic dataset structure so it can plug directly
    into TerraTorch's fine-tuning pipeline.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    import rasterio

    class MineralProspectivityDataset(Dataset):
        """
        Binary segmentation dataset for mineral deposit prediction.

        Each sample is a (image, mask) pair where:
          - image: (6, 224, 224) float32 tensor, normalized with Prithvi stats
          - mask: (224, 224) long tensor, 0=background, 1=deposit zone
        """

        def __init__(self, chips_dir: Path, labels_dir: Path, normalize: bool = True):
            self.chips_dir = chips_dir
            self.labels_dir = labels_dir
            self.normalize = normalize

            # Discover all chip files
            self.chip_files = sorted(chips_dir.glob("*.tif"))
            if len(self.chip_files) == 0:
                raise FileNotFoundError(
                    f"No chip files found in {chips_dir}. Run --prepare-data first."
                )

            # Pre-compute normalization tensors
            self.means = torch.tensor(PRITHVI_MEANS, dtype=torch.float32).view(6, 1, 1)
            self.stds = torch.tensor(PRITHVI_STDS, dtype=torch.float32).view(6, 1, 1)

        def __len__(self):
            return len(self.chip_files)

        def __getitem__(self, idx):
            chip_path = self.chip_files[idx]
            label_path = self.labels_dir / chip_path.name

            # Read image chip
            with rasterio.open(chip_path) as src:
                image = src.read().astype(np.float32)  # (6, H, W)

            # Read label mask
            with rasterio.open(label_path) as src:
                mask = src.read(1).astype(np.int64)  # (H, W)

            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

            # Normalize with Prithvi's HLS statistics
            # The model expects data normalized as: (pixel - mean) / std
            if self.normalize:
                image = (image - self.means) / self.stds

            return {"image": image, "mask": mask}

    # Build dataset
    dataset = MineralProspectivityDataset(CHIPS_DIR, LABELS_DIR)
    print(f"\nDataset: {len(dataset)} chips")

    # Count class balance
    n_positive = 0
    n_total = 0
    for i in range(min(len(dataset), 50)):  # sample first 50 for speed
        sample = dataset[i]
        n_positive += sample["mask"].sum().item()
        n_total += sample["mask"].numel()

    pos_ratio = n_positive / n_total if n_total > 0 else 0
    print(f"Positive pixel ratio (sampled): {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")
    print(f"Class imbalance ratio: 1:{int(1/pos_ratio) if pos_ratio > 0 else 'inf'}")

    # Train/val split (spatial split is better, but this works for prototyping)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    print(f"Train: {n_train} chips, Val: {n_val} chips")

    return train_loader, val_loader, dataset


# ===========================================================================
# Section 5: Fine-Tuning with TerraTorch (Recommended Approach)
# ===========================================================================

def generate_terratorch_config() -> Path:
    """
    Generate a TerraTorch YAML config file for fine-tuning Prithvi-EO-2.0
    on our mineral prospectivity segmentation task.

    TerraTorch is the official toolkit for Prithvi fine-tuning. It wraps
    PyTorch Lightning and provides pre-built model factories that combine
    Prithvi backbones with segmentation decoders (UNet, FPN, etc.).

    This is the RECOMMENDED approach for fine-tuning Prithvi.
    """
    config = {
        # ---------------------------------------------------------------
        # Trainer settings (PyTorch Lightning)
        # ---------------------------------------------------------------
        "trainer": {
            "max_epochs": 50,
            "accelerator": "auto",           # auto-detect GPU/MPS/CPU
            "devices": 1,
            "precision": "16-mixed",          # mixed precision for speed + memory
            "log_every_n_steps": 10,
            "default_root_dir": str(DATA_DIR / "outputs" / "foundation_model"),
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                    "init_args": {
                        "monitor": "val_loss",
                        "patience": 10,
                        "mode": "min",
                    },
                },
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "monitor": "val_loss",
                        "mode": "min",
                        "save_top_k": 3,
                        "filename": "prithvi-mineral-{epoch:02d}-{val_loss:.4f}",
                    },
                },
            ],
        },

        # ---------------------------------------------------------------
        # Data configuration (TerraTorch GenericNonGeoSegmentationDataModule)
        # ---------------------------------------------------------------
        "data": {
            "class_path": "terratorch.datamodules.GenericNonGeoSegmentationDataModule",
            "init_args": {
                "batch_size": 4,
                "num_workers": 4,
                # Paths to chip directories
                "train_data_root": str(CHIPS_DIR),
                "val_data_root": str(CHIPS_DIR),
                "test_data_root": str(CHIPS_DIR),
                # How to find images and labels
                "img_grep": "*_chip_*.tif",
                "label_grep": "*_chip_*.tif",
                "train_label_data_root": str(LABELS_DIR),
                "val_label_data_root": str(LABELS_DIR),
                "test_label_data_root": str(LABELS_DIR),
                # Split ratios (if no split files provided)
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
                # Band selection (6 Prithvi bands)
                "means": PRITHVI_MEANS,
                "stds": PRITHVI_STDS,
                "num_classes": 2,
            },
        },

        # ---------------------------------------------------------------
        # Model configuration
        # ---------------------------------------------------------------
        "model": {
            "class_path": "terratorch.tasks.SemanticSegmentationTask",
            "init_args": {
                # Backbone: Prithvi-EO-2.0-300M with temporal/location embeddings
                "model_args": {
                    "backbone": "prithvi_eo_v2_300",
                    "backbone_pretrained": True,
                    "backbone_bands": PRITHVI_BAND_NAMES,
                    "backbone_img_size": CHIP_SIZE,
                    "backbone_num_frames": 1,       # single temporal frame
                    "decoder": "UperNetDecoder",     # good for dense prediction
                    "num_classes": 2,
                    # Freeze backbone initially -- train only decoder
                    # Unfreeze after 5 epochs for full fine-tuning
                    "freeze_backbone": False,
                },
                # Loss: weighted cross-entropy to handle class imbalance
                # Our deposit pixels are ~1% of total, so upweight them heavily
                "loss": "ce",
                "class_weights": [1.0, 50.0],       # [background, deposit]
                # Optimizer
                "optimizer": "AdamW",
                "lr": 1e-4,                          # low LR for fine-tuning
                "weight_decay": 0.05,
                # LR scheduler
                "scheduler": "CosineAnnealingLR",
                "scheduler_kwargs": {"T_max": 50},
            },
        },
    }

    import yaml
    config_path = PROJECT_ROOT / "configs" / "prithvi_mineral_segmentation.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nTerraTorch config written to: {config_path}")
    print("\nTo fine-tune with TerraTorch CLI:")
    print(f"  terratorch fit --config {config_path}")
    print("\nTo run inference:")
    print(f"  terratorch predict --config {config_path} --ckpt_path <best_checkpoint.ckpt>")

    return config_path


# ===========================================================================
# Section 6: Manual Fine-Tuning Loop (Alternative -- No TerraTorch Required)
# ===========================================================================

def train_manual():
    """
    Manual fine-tuning loop using raw PyTorch.

    Use this if TerraTorch installation is problematic (e.g., GDAL issues
    on macOS). This gives you full control over the training process.

    Architecture: Prithvi-EO-2.0 encoder + simple segmentation head.
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # --- Device selection ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (this will be slow)")

    # --- Load the Prithvi backbone via timm ---
    # TerraTorch registers Prithvi models in the timm registry, but we can
    # also load directly from HuggingFace using the transformers library.
    try:
        import timm
        # If terratorch is installed, Prithvi models are registered in timm
        import terratorch  # noqa: F401 -- registers models as side effect

        encoder = timm.create_model(
            "prithvi_eo_v2_300",
            pretrained=True,
            num_frames=1,
            img_size=CHIP_SIZE,
            bands=PRITHVI_BAND_NAMES,
            features_only=True,           # output intermediate features for segmentation
        )
        # Get feature dimensions from the model
        dummy_input = torch.randn(1, 6, CHIP_SIZE, CHIP_SIZE)
        features = encoder(dummy_input)
        feature_dim = features[-1].shape[1]  # channels in last feature map
        print(f"Prithvi encoder loaded. Last feature dim: {feature_dim}")

    except (ImportError, RuntimeError) as e:
        print(f"Could not load Prithvi via timm/terratorch: {e}")
        print("Falling back to a ViT encoder from timm (not pre-trained on EO data)")
        print("Install terratorch for proper Prithvi support.\n")

        import timm
        # Use a generic ViT as placeholder -- same architecture family
        encoder = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            in_chans=6,
            features_only=False,
        )
        feature_dim = encoder.embed_dim
        print(f"Fallback ViT encoder loaded. Feature dim: {feature_dim}")

    # --- Simple segmentation head ---
    # For a production system, use UperNet or FPN. This is a minimal decoder
    # that upsamples encoder features to the input resolution.
    class SimpleSegmentationHead(nn.Module):
        """Minimal decoder: project features + bilinear upsample."""

        def __init__(self, in_dim: int, num_classes: int = 2):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(in_dim, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )

        def forward(self, x):
            return self.head(x)

    class PrithviSegmentor(nn.Module):
        """Prithvi encoder + segmentation head."""

        def __init__(self, backbone, seg_head, target_size: int = CHIP_SIZE):
            super().__init__()
            self.backbone = backbone
            self.seg_head = seg_head
            self.target_size = target_size

        def forward(self, x):
            # Get features from backbone
            features = self.backbone(x)
            if isinstance(features, list):
                feat = features[-1]  # use deepest feature map
            else:
                feat = features

            # If feat is (B, seq_len, dim) from ViT, reshape to spatial
            if feat.dim() == 3:
                B, N, C = feat.shape
                h = w = int(N ** 0.5)
                feat = feat.transpose(1, 2).view(B, C, h, w)

            # Decode
            logits = self.seg_head(feat)

            # Upsample to input resolution
            logits = nn.functional.interpolate(
                logits, size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False
            )
            return logits

    seg_head = SimpleSegmentationHead(feature_dim, num_classes=2)
    model = PrithviSegmentor(encoder, seg_head).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Data ---
    train_loader, val_loader, _ = build_dataset_and_dataloader()

    # --- Loss with class weights for extreme imbalance ---
    # Deposits are ~1% of pixels, so we upweight class 1 heavily
    class_weights = torch.tensor([1.0, 50.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer: low LR for fine-tuning, higher for new head ---
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},     # backbone: very low LR
        {"params": model.seg_head.parameters(), "lr": 1e-4},     # head: normal LR
    ], weight_decay=0.05)

    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    # --- Training loop ---
    num_epochs = 30
    best_val_loss = float("inf")

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("-" * 50)

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)                     # (B, 2, H, W)
            loss = criterion(logits, masks)            # masks: (B, H, W) long
            loss.backward()

            # Gradient clipping to stabilize fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                val_correct += (preds == masks).sum().item()
                val_total += masks.numel()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # --- Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_dir = DATA_DIR / "outputs" / "foundation_model" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "best_prithvi_mineral.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_loss={avg_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_dir / 'best_prithvi_mineral.pt'}")


# ===========================================================================
# Section 7: Clay Foundation Model (Alternative to Prithvi)
# ===========================================================================

def clay_overview():
    """
    Print an overview of how to use Clay Foundation Model as an alternative.

    Clay is a MAE-based model by Development Seed. Key differences from Prithvi:
      - Supports multiple sensors natively (Sentinel-2, Landsat, Sentinel-1, NAIP, MODIS)
      - Uses all 10 Sentinel-2 bands (vs Prithvi's 6 HLS bands)
      - Outputs embeddings rather than pixel-level features
      - Requires different normalization (sensor-specific stats in metadata.yaml)
      - Apache-2.0 license (vs Prithvi's Apache-2.0)

    For our mineral prospectivity task, Prithvi is likely the better choice because:
      1. TerraTorch provides a complete fine-tuning pipeline
      2. Prithvi has more segmentation examples and documentation
      3. The 300M-TL variant has location embeddings (useful for transfer)

    However, Clay's multi-sensor support could be valuable if we add SAR data later.
    """
    info = """
    ================================================================
    Clay Foundation Model -- Quick Reference
    ================================================================

    Model:      Clay v1.5 (MAE-based Vision Transformer)
    Org:        Development Seed / Clay Foundation
    License:    Apache-2.0
    HuggingFace: made-with-clay/Clay
    GitHub:     github.com/Clay-foundation/model
    Docs:       clay-foundation.github.io/model/

    Installation:
      pip install git+https://github.com/Clay-foundation/model.git

    Sentinel-2 bands supported (all 10):
      B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12

    Input format:
      - Batch dictionary with keys: pixels, timestep, latlon, gsd
      - pixels: normalized using sensor-specific stats from metadata.yaml
      - timestep: day-of-year encoding
      - latlon: geographic coordinates for spatial context
      - gsd: ground sample distance in meters

    Typical usage for embeddings:
      from clay.model import CLAYModule
      model = CLAYModule.load_from_checkpoint("path/to/clay-v1.5.ckpt")
      embeddings = model.encoder(batch)

    Fine-tuning for segmentation:
      Clay can be fine-tuned by adding a segmentation decoder on top of
      the encoder outputs. See Clay docs for examples:
      clay-foundation.github.io/model/tutorials/

    For our project:
      - Use Prithvi + TerraTorch first (more mature pipeline)
      - Consider Clay if we add Sentinel-1 SAR data for structural mapping
      - Clay embeddings could also be used as features for traditional ML
    ================================================================
    """
    print(info)


# ===========================================================================
# Section 8: Spatial Cross-Validation Note
# ===========================================================================

def print_spatial_cv_strategy():
    """
    Print guidance on implementing spatial cross-validation with foundation models.

    This is critical because our Phase 1 failure was spatial leakage.
    """
    print("""
    ================================================================
    Spatial Cross-Validation Strategy for Foundation Models
    ================================================================

    Phase 1 showed that Leave-One-Tile-Out (LOTO) PR-AUC dropped to 0.228.
    The foundation model approach helps because:

    1. PRE-TRAINED REPRESENTATIONS
       Prithvi learned spectral-spatial features from 4.2M global samples.
       These features are more invariant to atmospheric / illumination
       differences between tiles than raw band ratios.

    2. RECOMMENDED VALIDATION APPROACH
       - Split chips by TILE, not randomly
       - Train on 3 tiles, validate on 1 (LOTO)
       - Report LOTO PR-AUC as the primary metric
       - Random splits are for debugging only

    3. HOW TO IMPLEMENT TILE-BASED SPLITS
       Chip filenames encode the source tile (e.g., T35KRU_chip_00224_00448.tif).
       Filter train/val loaders by tile prefix:

         train_chips = [c for c in all_chips if not c.name.startswith(holdout_tile)]
         val_chips   = [c for c in all_chips if c.name.startswith(holdout_tile)]

    4. WHAT SUCCESS LOOKS LIKE
       - LOTO PR-AUC >= 0.50: Foundation model adds real value
       - LOTO PR-AUC >= 0.65: Cross-tile transfer is working
       - LOTO PR-AUC >= 0.75: Publishable result

    5. IF THE FOUNDATION MODEL ALSO FAILS LOTO
       - Try adding temporal data (multi-date composites)
       - Add DEM / terrain features as extra channels
       - Consider the 600M model (more capacity)
       - Investigate per-tile normalization before chipping
    ================================================================
    """)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Foundation model setup for mineral prospectivity mapping"
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Only check dependencies and download weights")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Prepare Sentinel-2 chips for Prithvi input format")
    parser.add_argument("--train", action="store_true",
                        help="Run fine-tuning (manual PyTorch loop)")
    parser.add_argument("--terratorch-config", action="store_true",
                        help="Generate TerraTorch YAML config for CLI-based training")
    parser.add_argument("--clay-info", action="store_true",
                        help="Print Clay Foundation Model overview")
    parser.add_argument("--cv-strategy", action="store_true",
                        help="Print spatial cross-validation guidance")
    parser.add_argument("--download-weights", action="store_true",
                        help="Download Prithvi model weights from HuggingFace")
    parser.add_argument("--model", type=str, default="300M-TL",
                        choices=["300M", "300M-TL", "600M"],
                        help="Which Prithvi variant to download (default: 300M-TL)")

    args = parser.parse_args()

    # If no flags, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\n--- Quick Start ---")
        print("1. Check dependencies:   python scripts/foundation_model_setup.py --check-only")
        print("2. Download weights:     python scripts/foundation_model_setup.py --download-weights")
        print("3. Prepare data chips:   python scripts/foundation_model_setup.py --prepare-data")
        print("4. Generate TT config:   python scripts/foundation_model_setup.py --terratorch-config")
        print("5. Train (manual loop):  python scripts/foundation_model_setup.py --train")
        print("\nOr use TerraTorch CLI:   terratorch fit --config configs/prithvi_mineral_segmentation.yaml")
        return

    # --- Dependency check (always runs) ---
    results = check_dependencies()
    all_ok = print_dependency_report(results)

    if args.check_only:
        return

    # --- Download weights ---
    if args.download_weights:
        model_map = {
            "300M": PRITHVI_300M_ID,
            "300M-TL": PRITHVI_300M_TL_ID,
            "600M": PRITHVI_600M_ID,
        }
        download_prithvi_weights(model_map[args.model])
        return

    # --- Clay info ---
    if args.clay_info:
        clay_overview()
        return

    # --- CV strategy ---
    if args.cv_strategy:
        print_spatial_cv_strategy()
        return

    # --- Prepare data ---
    if args.prepare_data:
        prepare_all_tiles()
        return

    # --- Generate TerraTorch config ---
    if args.terratorch_config:
        generate_terratorch_config()
        return

    # --- Train ---
    if args.train:
        if not all_ok:
            print("\nWARNING: Not all dependencies are installed.")
            print("Training may fail. Install missing packages first.")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                return
        train_manual()


if __name__ == "__main__":
    main()
