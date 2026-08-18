import torch
import torch.nn as nn

try:
    import lpips
except ImportError:
    lpips = None


class GeneratorLoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.bce = nn.BCEWithLogitsLoss()
        if lpips is not None:
            self.lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
        else:
            self.lpips_fn = None

    def forward(self, pred_raw, target, pred_real_d, pred_fake_d):
        charb = torch.sqrt((pred_raw - target) ** 2 + 1e-6).mean()
        fft = (torch.fft.rfft2(pred_raw).abs() - torch.fft.rfft2(target).abs()).abs().mean()

        pred_clamped = pred_raw.clamp(0, 1)
        target_clamped = target.clamp(0, 1)

        if self.lpips_fn is not None:
            perc = self.lpips_fn(
                (pred_clamped.repeat(1, 3, 1, 1) * 2) - 1,
                (target_clamped.repeat(1, 3, 1, 1) * 2) - 1,
            ).mean()
        else:
            perc = torch.zeros(1, device=self.device)

        real_d_mean = pred_real_d.mean().detach()
        fake_d_mean = pred_fake_d.mean().detach()

        loss_g_real = self.bce(pred_real_d - fake_d_mean, torch.zeros_like(pred_real_d))
        loss_g_fake = self.bce(pred_fake_d - real_d_mean, torch.ones_like(pred_fake_d))
        adv_loss = (loss_g_real + loss_g_fake) / 2.0

        total = (
            self.cfg['charb_w'] * charb +
            self.cfg['lpips_w'] * perc +
            self.cfg['fft_w'] * fft +
            self.cfg['adv_w'] * adv_loss
        )

        return total, {'charb': float(charb.item()), 'lpips': float(perc.item()), 'adv': float(adv_loss.item())}
