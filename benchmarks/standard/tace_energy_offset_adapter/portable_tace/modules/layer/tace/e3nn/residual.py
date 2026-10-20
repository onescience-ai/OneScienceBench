# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################

# import math
# import torch
# import torch.nn.functional as F
# from e3nn import o3


# from ..layout import LayoutTransform
# from .base import Residual
# from .linear import Linear, ElementLinear


# class AttentionResidual_BB(Residual):
#     def _setup(self):

#         self.alpha = 1 / math.sqrt(self.num_channel) # not use rms_norm for key
#         self.query = torch.nn.Parameter(torch.zeros(1, self.num_channel)) # if need, can add element
#         self.linear = torch.nn.ModuleList()
#         self.reshape = LayoutTransform(self.irreps_out)

#         irreps_in = [o3.Irreps(f"{self.num_channel}x0e")]
#         for _ in range(self.window):
#             irreps_in.append(self.irreps_in)
#         for idx in range(self.window):
#             if self.linear_type == 'aware':
#                 self.linear.append(
#                     ElementLinear(
#                         irreps_in[idx],
#                         self.irreps_out,
#                         bias=self.use_bias,
#                         num_elements=self.num_elements,
#                     )
#                 )
#             else:
#                 self.linear.append(
#                     Linear(
#                         irreps_in[idx],
#                         self.irreps_out,
#                         bias=self.use_bias,
#                     )
#                 )

#     def forward(self, prev_feats: list[torch.Tensor], node_attrs: torch.Tensor):

#         prev_feats = prev_feats[-self.window:]
#         key = torch.stack([feats[:, :self.num_channel] for feats in prev_feats], dim=0)
#         logits = torch.einsum('c, lbc -> lb', self.query.squeeze(0), key) * self.alpha
#         attn = F.softmax(logits, dim=0)
#         new_feats = []
#         for idx in range(self.window):
#             if self.linear_type == 'aware':
#                 new_feats.append(
#                     self.reshape(self.linear[idx](prev_feats[idx], node_attrs))
#                 )
#             else:
#                 new_feats.append(
#                     self.reshape(self.linear[idx](prev_feats[idx]))
#                 )
#         value = torch.stack(new_feats, dim=0)

#         return self.reshape.inverse(torch.einsum('lb, lbmc -> bmc', attn, value))


# class AttentionResidual_AB(Residual):
#     def _setup(self):

#         self.alpha = 1 / math.sqrt(self.num_channel) # not use rms_norm for key
#         self.query = torch.nn.Parameter(torch.zeros(1, self.num_channel)) # if need, can add element
#         self.linear = torch.nn.ModuleList()
#         self.reshape = LayoutTransform(self.irreps_out)

#         for idx in range(self.window):
#             if self.linear_type == 'aware':
#                 self.linear.append(
#                     ElementLinear(
#                         self.irreps_in,
#                         self.irreps_out,
#                         bias=self.use_bias,
#                         num_elements=self.num_elements,
#                     )
#                 )
#             else:
#                 self.linear.append(
#                     Linear(
#                         self.irreps_out,
#                         self.irreps_out,
#                         bias=self.use_bias,
#                     )
#                 )

#     def forward(self, prev_feats: list[torch.Tensor], node_attrs: torch.Tensor):

#         prev_feats = prev_feats[-self.window:]
#         key = torch.stack([feats[:, :self.num_channel] for feats in prev_feats], dim=0)
#         logits = torch.einsum('c, lbc -> lb', self.query.squeeze(0), key) * self.alpha
#         attn = F.softmax(logits, dim=0)
#         new_feats = []
#         for idx in range(self.window):
#             if self.linear_type == 'aware':
#                 new_feats.append(
#                     self.reshape(self.linear[idx](prev_feats[idx], node_attrs))
#                 )
#             else:
#                 new_feats.append(
#                     self.reshape(self.linear[idx](prev_feats[idx]))
#                 )
#         value = torch.stack(new_feats, dim=0)

#         return self.reshape.inverse(torch.einsum('lb, lbmc -> bmc', attn, value))


# class AttentionResidual(Residual):
#     def _setup(self):

#         self.alpha = 1 / math.sqrt(self.num_channel) # not use rms_norm for key
#         self.query = torch.nn.Parameter(torch.zeros(1, self.num_channel)) # if need, can add element
#         self.reshape = LayoutTransform(self.irreps_out)

#     def forward(self, prev_feats: list[torch.Tensor]):

#         prev_feats = prev_feats[-self.window:]
#         key = torch.stack([feats[:, :self.num_channel] for feats in prev_feats], dim=0)
#         logits = torch.einsum('c, lbc -> lb', self.query.squeeze(0), key) * self.alpha
#         attn = F.softmax(logits, dim=0)
#         value = torch.stack(prev_feats[-self.window:], dim=0)

#         return self.reshape.inverse(torch.einsum('lb, lbmc -> bmc', attn, value))


################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3

from ..linear import e3nnElementLinear, e3nnLinear


def get_resnet_layer(
    irreps_in,
    irreps_out,
    bias,
    num_elements,
    resnet_type,
):
    if resnet_type == "agnostic":
        return e3nnLinear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            bias=bias,
        )
    elif resnet_type == "identity":
        return SkipIdentity(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

    return e3nnElementLinear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        bias=bias,
        num_elements=num_elements,
    )


class ProjectUp(torch.nn.Module):
    """From https://github.com/SamsungDS/GGNN/blob/main/GGNN/model/EquFlashV2/nn/skip.py"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        idx = self._get_indices(self.irreps_in, self.irreps_out)
        self.register_buffer("idx", idx, persistent=False)

    @staticmethod
    def _get_indices(irreps_in, irreps_out):
        idxs = []
        out_slices = list(irreps_out.slices())
        used = set()

        for mul_ir in irreps_in:
            for index, (target, target_slice) in enumerate(zip(irreps_out, out_slices)):
                if index not in used and mul_ir == target:
                    idxs.append(torch.arange(target_slice.start, target_slice.stop))
                    used.add(index)
                    break
            else:
                raise ValueError(
                    f"{irreps_in} is not a sub-representation of {irreps_out}"
                )

        return torch.cat(idxs)

    def forward(self, x):
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        out[:, self.idx] = x
        return out


class ProjectDown(torch.nn.Module):
    """From https://github.com/SamsungDS/GGNN/blob/main/GGNN/model/EquFlashV2/nn/skip.py"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        idx = ProjectUp._get_indices(self.irreps_out, self.irreps_in)
        self.register_buffer("idx", idx, persistent=False)

    def forward(self, x):
        return x[:, self.idx]


class SkipIdentity(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        if self.irreps_in == self.irreps_out:
            self.proj = torch.nn.Identity()
        elif self.irreps_in.dim < self.irreps_out.dim:
            self.proj = ProjectUp(self.irreps_in, self.irreps_out)
        else:
            self.proj = ProjectDown(self.irreps_in, self.irreps_out)

    def forward(self, x):
        return self.proj(x)
