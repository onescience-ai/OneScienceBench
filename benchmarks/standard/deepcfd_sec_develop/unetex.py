import torch
import torch.nn as nn
import torch.nn.functional as F


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
        del bilinear
        self.inc = DoubleConv2D(
            in_channels,
            base_channels,
            normtype=normtype,
            kernel_size=kernel_size,
        )
        self.down_stages = nn.ModuleList()
        current_channels = base_channels
        for _ in range(num_stages):
            next_channels = current_channels * 2
            self.down_stages.append(
                Down2D(
                    current_channels,
                    next_channels,
                    normtype=normtype,
                    kernel_size=kernel_size,
                )
            )
            current_channels = next_channels

    def forward(self, x):
        features = [self.inc(x)]
        for stage in self.down_stages:
            features.append(stage(features[-1]))
        return features


class UNetDecoder2D(nn.Module):
    """Decode a contiguous range of U-Net levels.

    start_level is the current feature level and end_level is the final level.
    For example, start_level=4 and end_level=2 executes the 4->3 and 3->2
    upsampling stages.
    """

    def __init__(
        self,
        base_channels=16,
        num_stages=2,
        bilinear=True,
        normtype="bn",
        kernel_size=3,
        start_level=None,
        end_level=0,
    ):
        super().__init__()
        start_level = num_stages if start_level is None else int(start_level)
        end_level = int(end_level)
        if not 0 <= end_level <= start_level <= num_stages:
            raise ValueError(
                "decoder levels must satisfy "
                f"0 <= end_level <= start_level <= {num_stages}, got "
                f"start_level={start_level}, end_level={end_level}"
            )

        self.start_level = start_level
        self.end_level = end_level
        self.levels = list(range(start_level, end_level, -1))
        feature_channels = [
            base_channels * (2**level) for level in range(num_stages + 1)
        ]
        self.up_stages = nn.ModuleList()
        for level in self.levels:
            in_channels = (
                feature_channels[level] + feature_channels[level - 1]
                if bilinear
                else feature_channels[level]
            )
            self.up_stages.append(
                Up2D(
                    in_channels,
                    feature_channels[level - 1],
                    bilinear=bilinear,
                    normtype=normtype,
                    kernel_size=kernel_size,
                )
            )

    def forward(self, x, features=None):
        # Backward-compatible full-decoder call: decoder(features).
        if features is None:
            features = x
            x = None
        if len(features) <= self.start_level:
            raise ValueError(
                f"expected at least {self.start_level + 1} encoder features, "
                f"got {len(features)}"
            )
        if x is None:
            x = features[self.start_level]
        for up_stage, level in zip(self.up_stages, self.levels):
            x = up_stage(x, features[level - 1])
        return x


class UNetHead2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


class DecoderPath(nn.Module):
    def __init__(
        self,
        base_channels,
        num_stages,
        bilinear,
        normtype,
        kernel_size,
        start_level=None,
    ):
        super().__init__()
        self.decoder = UNetDecoder2D(
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size,
            start_level=start_level,
            end_level=0,
        )
        self.head = UNetHead2D(
            in_channels=base_channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, shared_feature, features=None):
        # Backward-compatible independent path call: decoder_path(features).
        if features is None:
            return self.head(self.decoder(shared_feature))
        return self.head(self.decoder(shared_feature, features))


class UNetEx(nn.Module):
    """U-Net with a shared encoder and partially shared decoder.

    The deepest ``shared_decoder_stages`` decoder stages are shared by all
    output variables. Remaining shallow stages and output heads are independent.
    Setting shared_decoder_stages=0 recovers the original independent-decoder
    topology; setting it to num_stages shares the complete decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        num_stages: int = 4,
        bilinear: bool = True,
        normtype: str = "bn",
        kernel_size: int = 3,
        shared_decoder_stages: int = 2,
    ):
        super().__init__()
        if num_stages < 1:
            raise ValueError("num_stages must be at least 1")
        if out_channels < 1:
            raise ValueError("out_channels must be at least 1")
        shared_decoder_stages = int(shared_decoder_stages)
        if not 0 <= shared_decoder_stages <= num_stages:
            raise ValueError(
                "shared_decoder_stages must satisfy "
                f"0 <= shared_decoder_stages <= {num_stages}, got "
                f"{shared_decoder_stages}"
            )

        self.num_stages = num_stages
        self.shared_decoder_stages = shared_decoder_stages
        self.branch_start_level = num_stages - shared_decoder_stages
        self.encoder = UNetEncoder2D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size,
        )
        self.shared_decoder = UNetDecoder2D(
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size,
            start_level=num_stages,
            end_level=self.branch_start_level,
        )
        self.decoders = nn.ModuleList(
            [
                DecoderPath(
                    base_channels=base_channels,
                    num_stages=num_stages,
                    start_level=self.branch_start_level,
                    bilinear=bilinear,
                    normtype=normtype,
                    kernel_size=kernel_size,
                )
                for _ in range(out_channels)
            ]
        )

    def forward(self, x):
        features = self.encoder(x)
        shared_feature = self.shared_decoder(features[-1], features)
        outputs = [
            decoder_path(shared_feature, features)
            for decoder_path in self.decoders
        ]
        return torch.cat(outputs, dim=1)
