"""Neural network architectures of the WFI-to-SWIR conditional GAN.

The :class:`UNetGenerator` is used for inference. The
:class:`PatchDiscriminator` is only needed for training and is kept here for
completeness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectionPadConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1,
                 use_in=True, activation='leaky', leaky_slope=0.2):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding=0)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True) if use_in else nn.Identity()
        if activation == 'leaky':
            self.act = nn.LeakyReLU(leaky_slope, inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return self.relu(out + identity)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Conv2d(F_int, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class UpBlockWithAtt(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, att, dropout_rate=0.0):
        super().__init__()
        self.att = att
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ReflectionPadConv(in_ch + skip_ch, out_ch, 3, 1, 1,
                                      use_in=True, activation='relu')
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.res = ResidualBlock(out_ch)
    def forward(self, x, skip):
        x_up = self.upsample(x)
        skip_att = self.att(x_up, skip)
        x_cat = torch.cat([x_up, skip_att], dim=1)
        x = self.conv(x_cat)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.res(x)
        return x

class UNetGenerator(nn.Module):
    """Attention U-Net that maps 4 WFI bands (B, G, R, NIR) to one SWIR band."""

    def __init__(self, dropout_rate=0.3, pad_input=16):
        super().__init__()
        self.pad_input = pad_input
        self.d1 = ReflectionPadConv(4, 128, 3, 2, 1, use_in=False, activation='leaky')
        self.r1 = ResidualBlock(128)
        self.d2 = ReflectionPadConv(128, 256, 3, 2, 1, use_in=True, activation='leaky')
        self.r2 = ResidualBlock(256)
        self.d3 = ReflectionPadConv(256, 512, 3, 2, 1, use_in=True, activation='leaky')
        self.r3 = ResidualBlock(512)
        self.d4 = ReflectionPadConv(512, 1024, 3, 2, 1, use_in=True, activation='leaky')
        self.r4 = ResidualBlock(1024)
        self.att3 = AttentionGate(1024, 512, 256)
        self.att2 = AttentionGate(512, 256, 128)
        self.att1 = AttentionGate(256, 128, 64)
        self.up1 = UpBlockWithAtt(1024, 512, 512, self.att3, dropout_rate=dropout_rate)
        self.up2 = UpBlockWithAtt(512, 256, 256, self.att2, dropout_rate=dropout_rate)
        self.up3 = UpBlockWithAtt(256, 128, 128, self.att1, dropout_rate=0.0)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ReflectionPadConv(128, 128, 3, 1, 1, use_in=True, activation='relu')
        )
        self.final = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x_pad = F.pad(x, (self.pad_input, self.pad_input, self.pad_input, self.pad_input),
                      mode='reflect')
        d1 = self.r1(self.d1(x_pad))
        d2 = self.r2(self.d2(d1))
        d3 = self.r3(self.d3(d2))
        d4 = self.r4(self.d4(d3))
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3)
        out = self.final(u4)
        _, _, h, w = x.shape
        return torch.sigmoid(out[:, :, self.pad_input:self.pad_input + h,
                                    self.pad_input:self.pad_input + w])

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator conditioned on the WFI input (training only)."""

    def __init__(self, in_channels=5):
        super().__init__()
        def block(i, o, s=2, norm=True):
            layers = [nn.Conv2d(i, o, 4, s, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64, 2, False),
            block(64, 128),
            block(128, 256),
            block(256, 512, 1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
