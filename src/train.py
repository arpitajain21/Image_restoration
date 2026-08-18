import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataset import build_dataloaders
from losses import GeneratorLoss
from nafnet_v5_gan import NAFNet_Generator, PatchDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Train NAFNet restoration model for semiconductor image enhancement.')
    parser.add_argument('--data_root', type=str, default='data/train/train', help='Root folder with GT and NoisyLR directories')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.005)
    parser.add_argument('--output_dir', type=str, default='models')
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    args = parse_args()
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'width': args.base_ch,
        'stage_blocks': 3,
        'mid_blocks': 6,
        'charb_w': 0.05,
        'lpips_w': 1.0,
        'fft_w': 0.1,
        'adv_w': 0.05,
        'lr_g': args.lr,
        'lr_d': args.lr,
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'speckle_prob': 0.5,
        'speckle_sigma_range': (0.01, 0.05),
        'gauss_noise_prob': 0.3,
        'gauss_sigma_range': (0.01, 0.04),
    }

    train_loader, val_loader, train_gt, val_gt = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_val_split=args.val_split,
        crop_size=args.patch_size,
        scale=2,
        num_workers=args.num_workers,
        seed=42,
    )

    netG = NAFNet_Generator(cfg).to(device)
    netD = PatchDiscriminator().to(device)

    optG = optim.AdamW(netG.parameters(), lr=cfg['lr_g'], betas=(0.9, 0.999))
    optD = optim.AdamW(netD.parameters(), lr=cfg['lr_d'], betas=(0.9, 0.999))

    criterionG = GeneratorLoss(cfg, device)
    criterionBCE = torch.nn.BCEWithLogitsLoss()
    psnr_calc = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    best_psnr = -1e9
    bad_epochs = 0
    history = {'epoch': [], 'd_loss': [], 'g_loss': [], 'val_psnr': [], 'val_ssim': []}

    for epoch in range(args.epochs):
        netG.train(); netD.train()
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0

        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device).float(), gt.to(device).float()

            optD.zero_grad()
            pred_real_d = netD(gt)
            with torch.no_grad():
                fake = netG(noisy).clamp(0, 1)
            pred_fake_d = netD(fake)

            loss_d_real = criterionBCE(pred_real_d - pred_fake_d.mean(), torch.ones_like(pred_real_d))
            loss_d_fake = criterionBCE(pred_fake_d - pred_real_d.mean(), torch.zeros_like(pred_fake_d))
            loss_D = (loss_d_real + loss_d_fake) / 2.0
            loss_D.backward()
            optD.step()

            optG.zero_grad()
            fake_raw = netG(noisy)
            pred_fake_d_for_g = netD(fake_raw.clamp(0, 1))
            with torch.no_grad():
                pred_real_d_for_g = netD(gt)
            loss_G, parts = criterionG(fake_raw, gt, pred_real_d_for_g, pred_fake_d_for_g)
            loss_G.backward()
            optG.step()

            d_loss_epoch += loss_D.item()
            g_loss_epoch += loss_G.item()

        netG.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                pred = torch.clamp(netG(noisy.to(device).float()), 0, 1)
                val_psnr += psnr_calc(pred, gt.to(device).float()).item()
                val_ssim += ssim_calc(pred, gt.to(device).float()).item()
        val_psnr /= max(1, len(val_loader))
        val_ssim /= max(1, len(val_loader))

        if val_psnr > best_psnr + args.min_delta:
            best_psnr = val_psnr
            np.save(os.path.join(args.output_dir, 'best_model.npy'), netG.state_dict(), allow_pickle=True)
            np.save(os.path.join(args.output_dir, 'best_discriminator.npy'), netD.state_dict(), allow_pickle=True)
            bad_epochs = 0
        else:
            bad_epochs += 1

        np.save(os.path.join(args.output_dir, 'last_model.npy'), netG.state_dict(), allow_pickle=True)

        history['epoch'].append(epoch + 1)
        history['d_loss'].append(d_loss_epoch / max(1, len(train_loader)))
        history['g_loss'].append(g_loss_epoch / max(1, len(train_loader)))
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        np.save(os.path.join(args.output_dir, 'train_log.npy'), history, allow_pickle=True)
        print(f'Epoch {epoch + 1}/{args.epochs} | D={history["d_loss"][-1]:.4f} | G={history["g_loss"][-1]:.4f} | PSNR={val_psnr:.3f} | SSIM={val_ssim:.3f} | Best={best_psnr:.3f}')

        if bad_epochs >= args.patience:
            print(f'Early stopping triggered at epoch {epoch + 1}.')
            break

    print(f'Finished. Best validation PSNR: {best_psnr:.3f}')


if __name__ == '__main__':
    train()
