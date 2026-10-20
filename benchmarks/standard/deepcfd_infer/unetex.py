import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderPath(nn.Module):
    def __init__(self, base_channels, num_stages, bilinear, normtype, kernel_size):
        super().__init__()
        self.decoder = UNetDecoder2D(
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size,
        )
        self.head = UNetHead2D(
            in_channels=base_channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, features):
        return self.head(self.decoder(features))


class UNetEx(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        num_stages: int = 2,
        bilinear: bool = True,
        normtype: str = "bn",
        kernel_size: int = 3,
    ):
        super().__init__()
        self.encoder = UNetEncoder2D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size,
        )
        self.decoders = nn.ModuleList(
            [
                DecoderPath(base_channels, num_stages, bilinear, normtype, kernel_size)
                for _ in range(out_channels)
            ]
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = [decoder_path(features) for decoder_path in self.decoders]
        return torch.cat(outputs, dim=1)


class UNetDecoder2D(nn.Module):
    def __init__(
        self,
        base_channels=16,
        num_stages=2,
        bilinear=True,
        normtype="bn",
        kernel_size=3,
    ):
        super().__init__()
        self.up_stages = nn.ModuleList()
        features = [base_channels * (2**i) for i in range(num_stages + 1)]

        for i in range(num_stages, 0, -1):
            in_channels = features[i] + features[i - 1] if bilinear else features[i]
            out_channels = features[i - 1]
            self.up_stages.append(
                Up2D(
                    in_channels,
                    out_channels,
                    bilinear,
                    normtype,
                    kernel_size=kernel_size,
                )
            )

    def forward(self, features):
        x = features[-1]
        for up_stage, skip in zip(self.up_stages, reversed(features[:-1])):
            x = up_stage(x, skip)
        return x

class UNetHead2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        return self.conv(x)
class DoubleConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        normtype="bn",
        kernel_size=3,
    ):
        super().__init__()
        mid_channels = mid_channels or out_channels
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2

        if normtype == "bn":
            first_norm = nn.BatchNorm2d(mid_channels)
            second_norm = nn.BatchNorm2d(out_channels)
        elif normtype == "in":
            first_norm = nn.InstanceNorm2d(mid_channels, affine=True)
            second_norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            first_norm = second_norm = None

        layers = [
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        ]
        if first_norm is not None:
            layers.append(first_norm)
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
            ]
        )
        if second_norm is not None:
            layers.append(second_norm)
        layers.append(nn.ReLU(inplace=True))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down2D(nn.Module):
    def __init__(self, in_channels, out_channels, normtype="bn", kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(
                in_channels,
                out_channels,
                normtype=normtype,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        normtype="bn",
        kernel_size=3,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            mid_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            mid_channels = None
        self.conv = DoubleConv2D(
            in_channels,
            out_channels,
            mid_channels,
            normtype=normtype,
            kernel_size=kernel_size,
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        height_diff = x2.size(2) - x1.size(2)
        width_diff = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [
                width_diff // 2,
                width_diff - width_diff // 2,
                height_diff // 2,
                height_diff - height_diff // 2,
            ],
        )
        return self.conv(torch.cat([x2, x1], dim=1))

class UNetEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels=16,
        num_stages=2,
        bilinear=True,
        normtype="bn",
        kernel_size=3,
    ):
        super().__init__()
        self.inc = DoubleConv2D(
            in_channels,
            base_channels,
            normtype=normtype,
            kernel_size=kernel_size,
        )
        self.down_stages = nn.ModuleList()
        in_channels = base_channels
        for _ in range(num_stages):
            out_channels = in_channels * 2
            self.down_stages.append(
                Down2D(
                    in_channels,
                    out_channels,
                    normtype=normtype,
                    kernel_size=kernel_size,
                )
            )
            in_channels = out_channels

    def forward(self, x):
        features = [self.inc(x)]
        for stage in self.down_stages:
            features.append(stage(features[-1]))
        return features
