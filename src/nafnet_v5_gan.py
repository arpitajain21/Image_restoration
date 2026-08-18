import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class SCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, groups=channels * 2)
        self.sg1 = SimpleGate()
        self.sca = SCA(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.ln2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 2, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        out = self.conv2(self.sca(self.sg1(self.dw(self.conv1(self.ln1(x))))))
        out = self.conv4(self.sg2(self.conv3(self.ln2(x + out))))
        return out + (x + out)


def make_stage(channels, count):
    return nn.Sequential(*[NAFBlock(channels) for _ in range(count)])


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


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.out = nn.Conv2d(ndf * 8, 1, 4, 1, 1)

    def forward(self, x):
        return self.out(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
