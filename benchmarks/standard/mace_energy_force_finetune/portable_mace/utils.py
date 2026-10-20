"""Small runtime helpers required by the vendored MACE interaction blocks."""

import torch


class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, data):
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, out, ctx.vec_len)
        return out, None
