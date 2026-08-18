# NAFNet V5 GAN — Joint Denoising & 2× Super-Resolution

Restores grayscale noisy low-resolution `.npy` images to clean, 2× upscaled
`.npy` images using a NAFNet-based generator fine-tuned with a PatchGAN /
RaGAN objective.

## Contents

```
team_name/
├── run.py            # inference entry point
├── requirements.txt  # dependencies (pinned)
├── README.md         # this file
└── models/
    └── nafnet_gan_finetuned_best.pth   # trained generator weights (see below)
```

## Setup

Requires Python 3.9+ and an NVIDIA GPU (CPU also works, just slower).

```bash
pip install -r requirements.txt
```

The correct CUDA build of PyTorch should be installed for the target machine.
If the default `torch==2.4.1` wheel does not match the local CUDA/driver, install
the matching build from https://pytorch.org and keep the same version, e.g.:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

No internet access, API keys, or additional downloads are needed at run time —
all weights are bundled in `models/`.

## Model weights

`run.py` loads, in order of preference:

1. `models/nafnet_gan_finetuned_best.pth` — the GAN fine-tuned generator.
2. `models/nafnet_gan_generator_best.pth` — the warm-start generator (fallback).

Place the trained `.pth` file in `models/` before running. These are the
`state_dict` files saved during training (`torch.save(netG.state_dict(), ...)`).

## Run

```bash
python run.py <input-dir> <output-dir>
```

- Reads every `.npy` file in `<input-dir>`.
- Creates `<output-dir>` if it does not exist.
- Writes one restored `.npy` per input, with the **same filename**.

### Input / output format

- **Input:** grayscale array, shape `(H, W)` or `(H, W, 1)`, low-resolution.
- **Output:** grayscale `float32` array, shape `(2H, 2W)` — the 2× target
  resolution — with values in `[0, 1]` and no `NaN`/`Inf`.

Inputs whose height or width is not a multiple of 4 are reflect-padded
internally and cropped back so the output is exactly `(2H, 2W)`.

## Model overview

- **Generator:** NAFNet encoder–decoder (width 64, 3 blocks/stage, 6 mid
  blocks) with a bilinear upsample skip (`base`) and a final PixelShuffle +
  refinement stage that performs the 2× super-resolution and suppresses
  checkerboard artifacts.
- **Training (reference only, not needed for inference):** Charbonnier + LPIPS
  + FFT-magnitude losses combined with a relativistic PatchGAN adversarial loss,
  warm-started from an earlier denoising checkpoint. Synthetic speckle and
  Gaussian noise were injected during training for robustness.
