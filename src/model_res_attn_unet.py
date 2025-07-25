"""
model.py - Contains Residual Attention UNet architecture for seismic velocity prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.ReLU(inplace=True)
        self.res_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        res = self.res_connection(x)
        return self.activation(out + res)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        return self.up(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv = ResidualConvBlock(in_channels, out_channels, dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.up = UpConv(in_channels, out_channels)
        self.att = attention_block(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.conv = ResidualConvBlock(in_channels, out_channels, dropout_rate)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.att(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class ResAttentionUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, dropout_rate=0.2):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, 64, dropout_rate)
        self.enc2 = EncoderBlock(64, 128, dropout_rate)
        self.enc3 = EncoderBlock(128, 256, dropout_rate)
        self.enc4 = EncoderBlock(256, 512, dropout_rate)

        self.bottleneck = ResidualConvBlock(512, 1024, dropout_rate)

        self.dec4 = DecoderBlock(1024, 512, dropout_rate)
        self.dec3 = DecoderBlock(512, 256, dropout_rate)
        self.dec2 = DecoderBlock(256, 128, dropout_rate)
        self.dec1 = DecoderBlock(128, 64, dropout_rate)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1, p1 = self.enc1(x)
        e2, p2 = self.enc2(p1)
        e3, p3 = self.enc3(p2)
        e4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.final(d1)

# Add EncoderBlock, DecoderBlock, and ResAttentionUNet classes here...
