import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) 模块。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """两层全连接 + GELU + Dropout 的通道混合模块。"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# AFNO 2D —— FourCastNet 核心空间混合模块
# ---------------------------------------------------------------------------

class AFNO2D(nn.Module):
    """Adaptive Fourier Neural Operator — 2D 频域空间混合。

    将输入 token 网格的每个通道在频域上分块，
    对各块独立做复数线性变换与软阈值收缩，
    最后逆变换回空间域。
    """

    def __init__(
        self,
        hidden_size,
        num_blocks=4,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_blocks ({num_blocks})"
            )
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction

        self.w1 = nn.Parameter(
            torch.randn(2, num_blocks, self.block_size, self.block_size)
            * (1.0 / math.sqrt(self.block_size))
        )
        self.b1 = nn.Parameter(
            torch.randn(2, num_blocks, self.block_size)
            * (1.0 / math.sqrt(self.block_size))
        )
        self.w2 = nn.Parameter(
            torch.randn(2, num_blocks, self.block_size, self.block_size)
            * (1.0 / math.sqrt(self.block_size))
        )
        self.b2 = nn.Parameter(
            torch.randn(2, num_blocks, self.block_size)
            * (1.0 / math.sqrt(self.block_size))
        )

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) —— token 网格
        Returns:
            (B, H, W, C) —— 频域混合后的 token 网格
        """
        B, H, W, C = x.shape

        # 转为频域表示: (B, H, W//2+1, C, 2)
        x = x.float()
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size, 1)
        x_r = x[..., 0].real
        x_i = x[..., 0].imag
        x = torch.stack([x_r, x_i], dim=-1)  # (B, H, W//2+1, num_blocks, block_size, 2)

        # 复数线性变换：o_real = w1[0] @ x_real - w1[1] @ x_imag + b1[0]
        o1_real = torch.einsum("...bk,bko->...bo", x[..., 0], self.w1[0])
        o1_real = o1_real - torch.einsum("...bk,bko->...bo", x[..., 1], self.w1[1])
        o1_real = o1_real + self.b1[0]

        o1_imag = torch.einsum("...bk,bko->...bo", x[..., 0], self.w1[1])
        o1_imag = o1_imag + torch.einsum("...bk,bko->...bo", x[..., 1], self.w1[0])
        o1_imag = o1_imag + self.b1[1]

        # 软阈值收缩
        o1_norm = torch.sqrt(o1_real ** 2 + o1_imag ** 2 + 1e-12)
        o1_norm = F.relu(o1_norm - self.sparsity_threshold)
        if self.hard_thresholding_fraction < 1.0:
            num_keep = max(
                1, int(self.hard_thresholding_fraction * o1_norm.shape[-1])
            )
            topk_vals, _ = torch.topk(o1_norm, num_keep, dim=-1)
            threshold = topk_vals[..., -1:]
            o1_norm = torch.where(o1_norm >= threshold, o1_norm, 0.0)

        scale = o1_norm / torch.sqrt(o1_real ** 2 + o1_imag ** 2 + 1e-12)
        o1_real = o1_real * scale
        o1_imag = o1_imag * scale

        # 复数线性变换 2
        o2_real = torch.einsum("...bk,bko->...bo", o1_real, self.w2[0])
        o2_real = o2_real - torch.einsum("...bk,bko->...bo", o1_imag, self.w2[1])
        o2_real = o2_real + self.b2[0]

        o2_imag = torch.einsum("...bk,bko->...bo", o1_real, self.w2[1])
        o2_imag = o2_imag + torch.einsum("...bk,bko->...bo", o1_imag, self.w2[0])
        o2_imag = o2_imag + self.b2[1]

        # 恢复复数 → 逆FFT
        o2 = torch.stack([o2_real, o2_imag], dim=-1)
        o2 = o2.reshape(B, H, W // 2 + 1, self.hidden_size, 2)
        o2 = torch.view_as_complex(o2)
        x = torch.fft.irfft2(o2, s=(H, W), dim=(1, 2), norm="ortho")

        return x.to(dtype=next(self.parameters()).dtype)


# ---------------------------------------------------------------------------
# FourCastNet 核心 Block
# ---------------------------------------------------------------------------

class FourCastNetBlock(nn.Module):
    """FourCastNet 的基础变换单元。

    结构：pre-norm + AFNO 频域空间混合 + 残差
         + pre-norm + MLP 通道混合 + 残差
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        num_blocks=4,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNO2D(
            hidden_size=dim,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden, drop=drop
        )

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) —— token 网格
        Returns:
            (B, H, W, C)
        """
        x = x + self.drop_path(self.filter(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# FourCastNet 完整模型
# ---------------------------------------------------------------------------

class FourCastNet(nn.Module):
    """FourCastNet 气象预测模型。

    流程：
    PatchEmbed(Conv2d) → PosEmbed → N×FourCastNetBlock(AFNO+MLP)
    → LinearHead → rearrange 恢复原始分辨率。

    Args:
        img_size:   输入空间尺寸 (H, W)
        patch_size: patch 切分尺寸 (Ph, Pw)
        in_chans:   输入气象变量通道数
        out_chans:  输出气象变量通道数
        embed_dim:  token 特征维度
        depth:      FourCastNetBlock 层数
        num_blocks: AFNO 频域分块数
        mlp_ratio:  MLP 隐藏层放大倍数
        drop_rate:  dropout 比例
        drop_path_rate: Stochastic Depth 最大比例
        sparsity_threshold: AFNO 软收缩阈值
        hard_thresholding_fraction: AFNO 频率保留比例
    """

    def __init__(
        self,
        img_size=(32, 64),
        patch_size=(4, 4),
        in_chans=20,
        out_chans=20,
        embed_dim=256,
        depth=4,
        num_blocks=4,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        self.patch_grid_h = img_size[0] // patch_size[0]
        self.patch_grid_w = img_size[1] // patch_size[1]
        num_patches = self.patch_grid_h * self.patch_grid_w

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic Depth 调度
        drop_path = np.linspace(0, drop_path_rate, depth).tolist()

        # AFNO Blocks
        self.blocks = nn.ModuleList([
            FourCastNetBlock(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path[i],
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            for i in range(depth)
        ])

        # 输出恢复头
        self.head = nn.Linear(
            embed_dim,
            out_chans * patch_size[0] * patch_size[1],
            bias=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 气象状态
        Returns:
            (B, out_chans, H, W) 预测气象状态
        """
        B = x.shape[0]

        x = self.patch_embed(x)                        # (B, C, H, W) → (B, E, h, w)
        x = x.flatten(2).transpose(1, 2)               # (B, N, E)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.reshape(B, self.patch_grid_h, self.patch_grid_w, self.embed_dim)

        for blk in self.blocks:
            x = blk(x)                                  # (B, h, w, E)

        x = self.head(x)                                # (B, h, w, out_chans*p1*p2)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.patch_grid_h,
            w=self.patch_grid_w,
        )
        return x


# ---------------------------------------------------------------------------
# 自回归推理包装器
# ---------------------------------------------------------------------------

class FourCastNetRollout(nn.Module):
    """基于 FourCastNet 的自回归序列推理模型。

    将 FourCastNet 包装为多步时间推进推理模型。
    每步用模型预测下一时刻气象状态，并将预测结果
    作为下一步输入。

    Args:
        img_size:       输入空间尺寸 (H, W)，默认 (32, 64)
        patch_size:     patch 切分尺寸，默认 (4, 4)
        in_chans:       输入变量通道数，默认 20
        out_chans:      输出变量通道数，默认 20
        embed_dim:      特征嵌入维度，默认 256
        depth:          主干 FourCastNetBlock 堆叠层数，默认 4
        num_blocks:     AFNO 频域分块数，默认 4
        mlp_ratio:      MLP 隐层放大倍数，默认 4.0
        drop_rate:      dropout 比例，默认 0.0
        drop_path_rate: Stochastic Depth 最大比例，默认 0.0
        sparsity_threshold:      AFNO 软收缩阈值，默认 0.01
        hard_thresholding_fraction: AFNO 频率保留比例，默认 1.0
    """

    def __init__(
        self,
        img_size=(32, 64),
        patch_size=(4, 4),
        in_chans=20,
        out_chans=20,
        embed_dim=256,
        depth=4,
        num_blocks=4,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.model = FourCastNet(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
        )

    def forward(self, x):
        """单步前向推理。

        Args:
            x: 当前气象状态 (B, C, H, W)

        Returns:
            下一时刻预测状态 (B, C, H, W)
        """
        return self.model(x)

    def autoregressive_rollout(self, initial_state, num_steps):
        """执行多步自回归推理。

        从初始状态出发，每步用 FourCastNet 预测下一时刻，
        并将预测结果作为下一步输入，循环 num_steps 次。

        Args:
            initial_state: 初始气象状态 (B, C, H, W)
            num_steps:     自回归推理步数

        Returns:
            trajectory: (num_steps, B, C, H, W)
        """
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}")

        trajectory = []
        current = initial_state

        for _ in range(num_steps):
            with torch.no_grad():
                current = self.forward(current)
            trajectory.append(current.clone())

        return torch.stack(trajectory, dim=0)

