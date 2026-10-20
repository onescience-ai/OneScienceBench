################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Union

import torch
from e3nn import o3
from e3nn.nn import Activation

from portable_tace.utils.tace.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import MLP
from .base import NodeEmbedding
from .fused import O3ScatterTensorProduct


class LinearNodeEmbedding(NodeEmbedding):
    """
    A simple node embedding module based on a linear transformation.

    This class projects discrete node attributes (e.g., element types)
    into a continuous feature space using a single linear layer,
    without introducing nonlinearity or structural information.
    """

    def _setup(self) -> None:

        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

        self.elem_emb1 = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        wigner,
        wigner_inv: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return self.elem_emb1(node_attrs)

    def __repr__(self) -> str:
        return repr(self.elem_emb1)


class NonLinearNodeEmbedding(NodeEmbedding):
    """
    A node embedding module with nonlinear transformation.

    This class applies a nonlinear activation function after a linear
    projection.
    """

    def _setup(self) -> None:

        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

        self.elem_emb1 = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.act1 = Activation(self.elem_emb1.irreps_out, [torch.nn.SiLU()])

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        wigner,
        wigner_inv: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        return self.act1(self.elem_emb1(node_attrs))


class TensorNodeEmbedding(NodeEmbedding):
    def _setup(self) -> None:

        self.node_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.source_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.target_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

        self.rejector = O3ScatterTensorProduct(
            [(self.num_channel, (0, 1))],
            [(1, (l, (-1) ** l)) for l in range(self.lmax + 1)],
            [(1, (l, (-1) ** l)) for l in range(self.Lmax + 1)],
        )

        self.irreps_out = self.rejector.irreps_out

        self.edge_info = MLP(
            channels=[
                self.num_radial_basis + self.num_channel * 2,
                self.num_channel,
                self.num_channel,
                self.rejector.weight_numel,
            ],
            bias=True,
            layer_norm=True,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        wigner,
        wigner_inv: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        base_node_feats = self.node_embedding(node_attrs)
        source_feats = self.source_embedding(node_attrs[edge_index[0]])
        target_feats = self.target_embedding(node_attrs[edge_index[1]])
        conv_weights = self.edge_info(
            torch.cat([edge_feats, source_feats, target_feats], dim=-1)
        )
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        node_feats = (
            self.rejector(
                torch.ones_like(base_node_feats),
                edge_attrs,
                conv_weights,
                edge_index,
            )
            / self.avg_num_neighbors
        )

        node_feats[:, : self.num_channel] = (
            node_feats.narrow(1, 0, self.num_channel) + base_node_feats
        )

        return node_feats


class SO2TensorNodeEmbedding(NodeEmbedding):
    def _setup(self) -> None:

        self.node_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.source_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        self.target_embedding = e3nnLinear(
            f"{self.num_elements}x0e", f"{self.num_channel}x0e", bias=self.bias
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)
        self.edge_info = MLP(
            channels=[
                self.num_radial_basis + self.num_channel * 2,
                self.num_channel,
                self.num_channel,
                (self.Lmax + 1) * self.num_channel,
            ],
            bias=True,
            layer_norm=True,
        )
        self.irreps_out = o3.Irreps(
            [(self.num_channel, (l, (-1) ** l)) for l in range(self.Lmax + 1)]
        )
        self.reshape = LayoutTransform(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        wigner,
        wigner_inv: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        base_node_feats = self.node_embedding(node_attrs)
        source_feats = self.source_embedding(node_attrs[edge_index[0]])
        target_feats = self.target_embedding(node_attrs[edge_index[1]])
        edge_feats = self.edge_info(
            torch.cat([edge_feats, source_feats, target_feats], dim=-1)
        )
        if cutoff is not None:
            edge_feats = edge_feats * cutoff

        edge_feats = edge_feats.view(
            edge_feats.size(0), (self.Lmax + 1), self.num_channel
        )
        edge_feats = torch.bmm(
            self.so2_angular_basis.wigner_inv.narrow(2, 0, (self.lmax + 1)), edge_feats
        )  # (edge, so3_m, C)

        node_feats = (
            scatter_sum(
                edge_feats, edge_index[1], dim=0, dim_size=base_node_feats.size(0)
            )
            / self.avg_num_neighbors
        )

        node_feats[:, 0:1, :] = node_feats.narrow(1, 0, 1) + base_node_feats.unsqueeze(
            1
        )

        return self.reshape.inverse(node_feats)


NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "nonlinear": NonLinearNodeEmbedding,
    "tensor": TensorNodeEmbedding,
    "so2_tensor": SO2TensorNodeEmbedding,
}
