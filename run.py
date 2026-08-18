#!/usr/bin/env python3
"""
run.py — Joint Denoising & 2x Super-Resolution inference (NAFNet V5 GAN generator).

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy file in <input-dir> (each a grayscale noisy low-res array),
restores it with the trained generator, and writes one .npy per input to
<output-dir> using the same filename. Outputs are float32 grayscale arrays
in [0, 1] with no NaN/Inf, at the 2x target resolution.
"""

import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Model configuration (must match the architecture the weights were trained on)
# ----------------------------------------------------------------------------
CFG = {
    'width': 64,
    'mid_blocks': 6,
    'stage_blocks': 3,
    'scale': 2,
}

# Weights file shipped inside models/. Falls back to the warm-start checkpoint
# if the fine-tuned one is not present.
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_CANDIDATES = [
    os.path.join(_HERE, 'models', 'nafnet_gan_finetuned_best.pth'),
    os.path.join(_HERE, 'models', 'nafnet_gan_generator_best.pth'),
]

# The network downsamples spatially by a factor of 4 (two stride-2 downs),
# so input H and W are padded up to a multiple of this before the forward pass.
PAD_MULTIPLE = 4


# -----------------------
# Generator architecture 
# -----------------------
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class SCA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2)
        self.sg1 = SimpleGate()
        self.sca = SCA(c)
        self.conv2 = nn.Conv2d(c, c, 1)
        self.ln2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, c * 2, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(c, c, 1)

    def forward(self, x):
        out = self.conv2(self.sca(self.sg1(self.dw(self.conv1(self.ln1(x))))))
        out = self.conv4(self.sg2(self.conv3(self.ln2(x + out))))
        return out + (x + out)


def make_stage(c, n):
    return nn.Sequential(*[NAFBlock(c) for _ in range(n)])


class NAFNet_Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w = cfg['width']
        nb = cfg['stage_blocks']
        mb = cfg['mid_blocks']

        self.intro = nn.Conv2d(1, w, 3, 1, 1)
        self.enc1 = make_stage(w, nb)
        self.down1 = nn.Conv2d(w, w * 2, 2, 2)
        self.enc2 = make_stage(w * 2, nb)
        self.down2 = nn.Conv2d(w * 2, w * 4, 2, 2)
        self.mid = make_stage(w * 4, mb)
        self.up1 = nn.Sequential(nn.Conv2d(w * 4, w * 8, 1), nn.PixelShuffle(2))
        self.dec1 = make_stage(w * 2, nb)
        self.up2 = nn.Sequential(nn.Conv2d(w * 2, w * 4, 1), nn.PixelShuffle(2))
        self.dec2 = make_stage(w, nb)

        self.up3 = nn.Sequential(nn.Conv2d(w, w * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.refinement = make_stage(w, 1)
        self.out_conv = nn.Conv2d(w, 1, 3, 1, 1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        e1 = self.enc1(self.intro(x))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        u1 = self.dec1(self.up1(m) + e2)
        u2 = self.dec2(self.up2(u1) + e1)
        out = self.out_conv(self.refinement(self.up3(u2)))
        return base + out


# ------------------
# Inference helpers
# ------------------
def load_model(device):
    weight_path = next((p for p in WEIGHT_CANDIDATES if os.path.exists(p)), None)
    if weight_path is None:
        raise FileNotFoundError(
            "No model weights found. Expected one of:\n  "
            + "\n  ".join(WEIGHT_CANDIDATES)
        )
    model = NAFNet_Generator(CFG).to(device)
    state = torch.load(weight_path, map_location=device)
    # Support both bare state_dicts and {'state_dict': ...} wrappers.
    if isinstance(state, dict) and 'state_dict' in state and \
            all(not k.startswith(('intro', 'enc', 'mid', 'dec', 'up', 'down', 'out', 'refinement'))
                for k in list(state.keys())[:1]):
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded weights: {weight_path}")
    return model


def to_2d(arr):
    """Squeeze an input array to 2-D (H, W)."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # (H, W, 1) or (1, H, W)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    return arr.astype(np.float32)


@torch.no_grad()
def restore(model, arr2d, device):
    h, w = arr2d.shape
    x = torch.from_numpy(arr2d).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    # Pad so H and W are multiples of PAD_MULTIPLE (reflect padding).
    pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
    pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    out = model(x)  # 2x upscale

    # Crop back to the true 2x target resolution (2H, 2W).
    out = out[:, :, : 2 * h, : 2 * w]
    out = out.squeeze(0).squeeze(0).float().cpu().numpy()

    # Sanitize: clamp to [0,1] and remove any NaN/Inf.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(device)

    files = sorted(glob.glob(os.path.join(input_dir, '*.npy')))
    if not files:
        print(f"No .npy files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} input file(s).")
    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        arr = to_2d(np.load(fpath))
        out = restore(model, arr, device)
        np.save(os.path.join(output_dir, fname), out)
        print(f"[{i}/{len(files)}] {fname}  {arr.shape} -> {out.shape}")

    print("Done.")


if __name__ == '__main__':
    main()
