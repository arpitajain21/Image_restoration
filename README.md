# AI-Based Restoration of Degraded Images for Semiconductor Inspection

NAFNet V5 GAN-based image restoration for denoising and 2x super-resolution of semiconductor wafer inspection images.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python run.py <input-dir> <output-dir>

# Example
python run.py data/test outputs
```

## Overview

Joint denoising and 2x super-resolution for grayscale semiconductor images using:
- **NAFNet V5**: State-of-the-art restoration architecture with normalized activation functions
- **GAN Generator**: Adversarial training for perceptually pleasing outputs
- **Loss Functions**: L1 reconstruction + LPIPS perceptual + adversarial loss

## Project Structure

```
image_enhancement/
├── README.md                           # This file
├── requirements.txt                    # Dependencies (numpy, torch, etc.)
├── run.py                              # Inference entry point
│
├── models/
│   └── nafnet_gan_finetuned_best.pth   # Pre-trained NAFNet generator weights
│
├── src/                                # Core module
│   ├── __init__.py
│   ├── nafnet_v5_gan.py               # NAFNet V5 architecture (NAFBlock, NAFNet_Generator, PatchDiscriminator)
│   ├── train.py                       # Training pipeline with loss computation
│   ├── inference.py                   # Inference utilities
│   ├── dataset.py                     # Dataset loading and preprocessing
│   ├── losses.py                      # Loss functions (L1, LPIPS, GAN)
│   └── __pycache__/                   # Python bytecode cache
│
├── data/
│   ├── train/
│   │   ├── GT/                        # Ground truth high-res (3200+ images)
│   │   │   └── 000000.npy to 003199.npy
│   │   └── NoisyLR/                   # Noisy low-res training inputs
│   │       └── 000000.npy to 003199.npy
│   └── test/
│       └── NoisyLR/                   # Test images for inference
│           └── 000000.npy to 000174.npy
│
├── outputs/                            # Inference results (auto-created)
│   └── 000000.npy, 000001.npy, ...    # Restored high-res outputs
│
├── notebooks/
│   ├── EDA.ipynb                      # Dataset exploration (01_EDA)
│   └── experiments/
│       ├── NAFNET.ipynb               # Initial NAFNet experiments
│       ├── NAFNETversion2.ipynb        # NAFNet V2 experiments
│       └── DnCNN.ipynb                # DnCNN baseline experiments
```

## Dependencies

```
numpy==2.2.6
scipy==1.16.0
torch==2.6.0
torchmetrics==1.8.2
lpips==0.1.4
```

Install: `pip install -r requirements.txt`

## Usage

### Inference (Submission)

```bash
python run.py <input-dir> <output-dir>
```

Reads `.npy` files from `<input-dir>`, restores them, and saves outputs to `<output-dir>`.

**Example:**
```bash
python run.py data/test outputs
```

### Training

```bash
python src/train.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --base_ch 32 \
  --patch_size 128 \
  --data_root data/train/train
```

## Model Architecture

**Yes, the model contains NAFNet!** 

The implementation includes:
- **`NAFBlock`** - Core building block with normalized activation functions and spatial channel attention
- **`NAFNet_Generator`** - Full generator using multi-scale NAFBlocks with encoder-decoder + 2x upsampling
- **`PatchDiscriminator`** - Patch-based discriminator for adversarial training

**NAFNet V5 Configuration:**
```python
{
  'width': 64,           # Base channels
  'mid_blocks': 6,       # Blocks in bottleneck
  'stage_blocks': 3,     # Blocks per stage
  'scale': 2             # 2x upsampling
}
```

**Architecture Details:**
- Input: 1-channel (grayscale) noisy LR image
- Encoder: 2 downsampling stages with NAFBlocks
- Bottleneck: 6 middle NAFBlocks at deepest resolution
- Decoder: 2 upsampling stages with PixelShuffle + NAFBlocks
- Output: 1-channel restored 2x HR image + residual learning

**Loss Functions:**
- **L1 Loss**: Pixel-level reconstruction
- **LPIPS Loss**: Learned perceptual image patch similarity
- **GAN Loss**: Adversarial training with PatchDiscriminator
- **Combined**: Weighted sum for balanced training

## Input/Output

| Aspect | Details |
|--------|---------|
| **Input** | Grayscale noisy low-res (LR) images in `.npy` format, float32, range [0, 1] |
| **Output** | Grayscale high-res (HR) images, 2x resolution, float32, range [0, 1] |
| **Data Format** | NumPy `.npy` binary arrays |
| **Processing** | Automatic padding (multiples of 4), clipping to valid range |

## Key Files

**Model Implementation:**
- `src/nafnet_v5_gan.py` - Contains NAFBlock, NAFNet_Generator, and PatchDiscriminator classes
  - `LayerNorm2d` - Channel-wise layer normalization
  - `SimpleGate` - Gating mechanism (x * sigmoid(x))
  - `SCA` - Spatial channel attention module
  - `NAFBlock` - NAF block with normalized activations
  - `NAFNet_Generator` - Full restoration network
  - `PatchDiscriminator` - Adversarial discriminator

**Training & Inference:**
- `src/train.py` - Training loop with multi-loss optimization
- `src/inference.py` - Batch inference utilities
- `src/losses.py` - L1, LPIPS, and GAN loss implementations
- `src/dataset.py` - DataLoader for `.npy` files

**Entry Points:**
- `run.py` - Production inference script (loads model and processes images)
- `notebooks/EDA.ipynb` - Dataset visualization and exploration

**Experiments:**
- `notebooks/experiments/` - Development notebooks for different architectures (NAFNET, DnCNN)
