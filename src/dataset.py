import glob
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KLADataset(Dataset):
    def __init__(self, gt_paths, noisy_paths, train=True, crop_size=128, scale=2, noise_config=None):
        self.gt_paths = list(gt_paths)
        self.noisy_paths = list(noisy_paths)
        self.train = train
        self.crop_size = crop_size
        self.scale = scale
        self.noise_config = noise_config or {
            'speckle_prob': 0.5,
            'speckle_sigma_range': (0.01, 0.05),
            'gauss_noise_prob': 0.3,
            'gauss_sigma_range': (0.01, 0.04),
        }

    def __len__(self):
        return len(self.gt_paths)

    def _apply_augmentations(self, noisy, gt):
        rng = np.random.default_rng()
        h_lr, w_lr = noisy.shape

        if h_lr > self.crop_size and w_lr > self.crop_size:
            x = int(rng.integers(0, w_lr - self.crop_size + 1))
            y = int(rng.integers(0, h_lr - self.crop_size + 1))
            noisy_crop = noisy[y:y + self.crop_size, x:x + self.crop_size]
            gt_crop = gt[y * self.scale:(y + self.crop_size) * self.scale, x * self.scale:(x + self.crop_size) * self.scale]
            noisy, gt = noisy_crop, gt_crop

        if rng.random() > 0.5:
            noisy, gt = np.fliplr(noisy).copy(), np.fliplr(gt).copy()
        if rng.random() > 0.5:
            noisy, gt = np.flipud(noisy).copy(), np.flipud(gt).copy()
        k = int(rng.integers(0, 4))
        if k:
            noisy, gt = np.rot90(noisy, k).copy(), np.rot90(gt, k).copy()

        if rng.random() < self.noise_config['speckle_prob']:
            noisy = noisy + noisy * rng.normal(0.0, float(rng.uniform(*self.noise_config['speckle_sigma_range'])), noisy.shape).astype(np.float32)
        if rng.random() < self.noise_config['gauss_noise_prob']:
            noisy = noisy + rng.normal(0.0, float(rng.uniform(*self.noise_config['gauss_sigma_range'])), noisy.shape).astype(np.float32)

        return noisy.astype(np.float32), gt.astype(np.float32)

    def __getitem__(self, idx):
        gt = np.load(self.gt_paths[idx]).astype(np.float32)
        noisy = np.load(self.noisy_paths[idx]).astype(np.float32)

        if self.train and noisy.shape[0] >= self.crop_size and noisy.shape[1] >= self.crop_size:
            noisy, gt = self._apply_augmentations(noisy, gt)

        return (
            torch.from_numpy(np.ascontiguousarray(noisy)).unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0),
        )


def make_split(gt_root: str, noisy_root: str, val_fraction: float = 0.1, seed: int = 42):
    gt_files = sorted(glob.glob(os.path.join(gt_root, '*.npy')))
    noisy_files = sorted(glob.glob(os.path.join(noisy_root, '*.npy')))

    pairs = list(zip(gt_files, noisy_files))
    random.seed(seed)
    random.shuffle(pairs)

    split = max(1, int(len(pairs) * (1.0 - val_fraction)))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    train_gt, train_noisy = zip(*train_pairs) if train_pairs else ([], [])
    val_gt, val_noisy = zip(*val_pairs) if val_pairs else ([], [])
    return list(train_gt), list(train_noisy), list(val_gt), list(val_noisy)


def build_dataloaders(data_root: str, batch_size: int = 8, train_val_split: float = 0.1, crop_size: int = 128, scale: int = 2, num_workers: int = 0, seed: int = 42):
    gt_root = os.path.join(data_root, 'GT')
    noisy_root = os.path.join(data_root, 'NoisyLR')

    train_gt, train_noisy, val_gt, val_noisy = make_split(gt_root, noisy_root, val_fraction=train_val_split, seed=seed)

    train_ds = KLADataset(train_gt, train_noisy, train=True, crop_size=crop_size, scale=scale)
    val_ds = KLADataset(val_gt, val_noisy, train=False, crop_size=crop_size, scale=scale)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_gt, val_gt


if __name__ == '__main__':
    dl, _, _, _ = build_dataloaders('data/train/train', batch_size=2)
    x, y = next(iter(dl))
    print(x.shape, y.shape)
