"""Paper-first GeoMPNN for the 2-D AirfRANS experiment (arXiv:2412.09399v2).

Priority rule used in this file:
1. If the paper explicitly specifies a formula/operation, implement it literally.
2. Only for details not disclosed by the paper, follow the official AIRS GeoMPNN code.

Consequences versus the public repository:
- Eq. (7) uses d = 4L/(s*pi) exactly and exponent i/n_basis.
- Eq. (9)-(10) use the paper's literal factorial normalization and P_l(sin(theta)) odd branch.
- theta^4 is ordered as rotations 0, +90, +180, +270 degrees.
- surface edge features use the literal displacement y-x (no division by surf_radius).
- Surf2Vol updates follow the paper equations directly; there is no extra independent
  volume-node processor between Surf2Vol layers.
- the decoder consumes only the final latent node representation z(x), as stated in Sec. 4.2.

Repository fallbacks retained because the paper does not disclose them include hidden size,
MLP widths, MLP normalization placement, and exact radius-neighbor random sampling machinery.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

VOLUME = "volume"
SURFACE = "surface"
SURFACE2VOLUME = "surf2vol"
NOSE = "nose"
TAIL = "tail"
CLOSEST = "closest"


def change_basis(coords: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Paper Sec. 4.5.1: apply R_v so inlet velocity becomes parallel to +x.

    Paper defines, for column vectors,
        R_v = 1/||v|| [[v1, v2], [-v2, v1]].
    For an (N,2) row-vector tensor we therefore multiply by R_v^T.
    """
    if basis.ndim != 1 or basis.numel() != 2:
        raise ValueError(f"basis must have shape (2,), got {tuple(basis.shape)}")
    v1, v2 = basis[0], basis[1]
    norm = basis.norm().clamp_min(eps)
    r_t = torch.stack([
        torch.stack([v1, -v2]),
        torch.stack([v2,  v1]),
    ]) / norm
    return coords @ r_t


def _segment_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    cnt = src.new_zeros((dim_size, 1))
    cnt.index_add_(0, index, torch.ones((len(index), 1), device=src.device, dtype=src.dtype))
    return out / cnt.clamp_min_(1.0)


class InstanceNorm1Graph(nn.Module):
    """Single-graph equivalent of PyG InstanceNorm(..., affine=True).

    This normalization detail is not specified in the paper and follows the official code.
    """
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(0, keepdim=True)
        var = (x - mean).pow(2).mean(0, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class ManualRadiusGraph(nn.Module):
    """Radius graph with no self loops and random truncation to M neighbors.

    Paper Sec. 4.2 specifies r=0.05 and M=8 for the final surface graph.
    The exact random truncation implementation is not disclosed; this mirrors the official
    implementation's per-target random selection behavior.
    """
    def __init__(self, r: float, max_num_neighbors: int):
        super().__init__()
        self.r = float(r)
        self.max_num_neighbors = int(max_num_neighbors)

    @torch.no_grad()
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        if self.r == 0:
            return torch.empty((2, 0), device=pos.device, dtype=torch.long)
        n = len(pos)
        d = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)
        d.fill_diagonal_(float("inf"))  # official graph behavior: no self loops
        sources, targets = [], []
        for target in range(n):
            cand = torch.where(d[target] <= self.r)[0]
            if cand.numel() > self.max_num_neighbors:
                perm = torch.randperm(cand.numel(), device=pos.device)[: self.max_num_neighbors]
                cand = cand[perm]
            sources.append(cand)
            targets.append(torch.full_like(cand, target))
        if not sources or sum(x.numel() for x in sources) == 0:
            return torch.empty((2, 0), device=pos.device, dtype=torch.long)
        return torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)


class SinusoidalEmbedding(nn.Module):
    """Literal paper Eq. (7).

    PE(x) = [sin((x/s)/d^(i/n)), cos((x/s)/d^(i/n))] for i=0..n-1,
    with d = 4L/(s*pi).

    The paper gives n_basis=8. Numerical s and L are not disclosed in the paper; config.yaml
    uses the official repository values s=0.001 and L=8 as fallbacks.
    """
    def __init__(self, n_basis: int, spacing_s: float, domain_size_l: float):
        super().__init__()
        if n_basis <= 0:
            raise ValueError("n_basis must be positive")
        if spacing_s <= 0 or domain_size_l <= 0:
            raise ValueError("spacing_s and domain_size_l must be positive")
        self.n_basis = int(n_basis)
        self.spacing_s = float(spacing_s)
        d = 4.0 * float(domain_size_l) / (self.spacing_s * math.pi)
        i = torch.arange(self.n_basis, dtype=torch.float32)
        freq = torch.pow(torch.tensor(d, dtype=torch.float32), -i / float(self.n_basis))
        self.register_buffer("freq", freq)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        n = len(coords)
        flat = coords.flatten() / self.spacing_s
        phase = flat.unsqueeze(1) * self.freq.unsqueeze(0)
        emb = torch.cat([phase.sin(), phase.cos()], dim=1)
        return emb.unflatten(0, (n, -1)).flatten(1)


def _legendre_sequence(x: torch.Tensor, max_l: int) -> torch.Tensor:
    """Return P_1(x), ..., P_max_l(x)."""
    if max_l == 0:
        return x.new_empty((*x.shape, 0))
    p0 = torch.ones_like(x)
    p1 = x
    vals = [p1]
    for ell in range(2, max_l + 1):
        p = ((2 * ell - 1) * x * p1 - (ell - 1) * p0) / ell
        vals.append(p)
        p0, p1 = p1, p
    return torch.stack(vals, dim=-1)


class SphericalHarmonicsEmbedding(nn.Module):
    """Literal paper Eq. (9)-(10), including the printed factorial.

    Y_l^0(theta)      = sqrt((2l+1)!/(4*pi)) P_l(cos(theta))
    Ybar_l^0(theta)   = sqrt((2l+1)!/(4*pi)) P_l(sin(theta))

    This intentionally differs from the public code, which uses the standard
    sqrt((2l+1)/(4*pi)) coefficient and evaluates its second branch via theta+pi/2.
    """
    def __init__(self, num_spherical: int):
        super().__init__()
        self.num_spherical = int(num_spherical)
        norms = [math.sqrt(math.factorial(2 * ell + 1) / (4.0 * math.pi))
                 for ell in range(1, self.num_spherical + 1)]
        self.register_buffer("norm", torch.tensor(norms, dtype=torch.float32))

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        theta = angle.reshape(-1)
        even = _legendre_sequence(torch.cos(theta), self.num_spherical) * self.norm
        odd = _legendre_sequence(torch.sin(theta), self.num_spherical) * self.norm
        return torch.cat([even, odd], dim=-1).reshape(len(angle), -1)


def azimuth4(coords: torch.Tensor) -> torch.Tensor:
    """Paper theta^4 order: x, R_90 x, R_180 x, R_270 x (positive CCW rotations)."""
    x, y = coords[:, 0], coords[:, 1]
    xx = torch.stack([x, -y, -x, y], dim=-1)
    yy = torch.stack([y,  x, -y, -x], dim=-1)
    return torch.atan2(yy, xx)


class EmbDist(nn.Module):
    def __init__(self, n_basis: int, spacing_s: float, domain_size_l: float):
        super().__init__()
        self.sine = SinusoidalEmbedding(n_basis, spacing_s, domain_size_l)
        self.dim = 1 + 2 * n_basis

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.cat([dist, self.sine(dist)], dim=-1)


class EmbCoords(nn.Module):
    def __init__(self, n_basis: int, spacing_s: float, domain_size_l: float, rotate: bool):
        super().__init__()
        self.sine = SinusoidalEmbedding(n_basis, spacing_s, domain_size_l)
        self.rotate = rotate
        self.dim = 2 + 4 * n_basis

    def forward(self, coords: torch.Tensor, basis: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rotate:
            if basis is None:
                raise ValueError("basis is required for inlet-canonicalized coordinates")
            coords = change_basis(coords, basis)
        return torch.cat([coords, self.sine(coords)], dim=-1)


class EmbAngle(nn.Module):
    def __init__(self, num_spherical: int, rotate: bool):
        super().__init__()
        self.sph = SphericalHarmonicsEmbedding(num_spherical)
        self.rotate = rotate
        self.dim = 8 * num_spherical  # 4 axes * (even + odd) * n_basis

    def forward(self, coords: torch.Tensor, basis: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rotate:
            if basis is None:
                raise ValueError("basis is required for inlet-canonicalized angles")
            coords = change_basis(coords, basis)
        return self.sph(azimuth4(coords)).reshape(len(coords), -1)


class MLP(nn.Module):
    """Two-linear-layer MLP used by the official implementation (paper does not specify depth)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, norm: Optional[str] = None):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.prenorm = InstanceNorm1Graph(input_dim) if norm == "pre" else None
        self.pre_act_norm = InstanceNorm1Graph(hidden_dim) if norm == "preAct" else None
        self.post_act_norm = InstanceNorm1Graph(hidden_dim) if norm == "postAct" else None
        self.postnorm = InstanceNorm1Graph(output_dim) if norm == "post" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm is not None:
            x = self.prenorm(x)
        x = self.lin1(x)
        if self.pre_act_norm is not None:
            x = self.pre_act_norm(x)
        x = F.relu(x)
        if self.post_act_norm is not None:
            x = self.post_act_norm(x)
        x = self.lin2(x)
        if self.postnorm is not None:
            x = self.postnorm(x)
        return x


class MessageProcessor(nn.Module):
    """Residual edge/node update matching the paper's message-passing equations."""
    def __init__(self, hidden_dim: int, mlp_norm: Optional[str], edge_norm: bool = False):
        super().__init__()
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, mlp_norm if edge_norm else None)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_norm)

    def forward(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        source_ind, target_ind = edge_index
        delta_e = self.edge_mlp(torch.cat([x_source[source_ind], x_target[target_ind], edge_attr], dim=-1))
        edge_attr = edge_attr + delta_e
        msg = _segment_mean(edge_attr, target_ind, len(x_target))
        x_target = x_target + self.node_mlp(torch.cat([x_target, msg], dim=-1))
        return x_target, edge_attr


class GeoMPNN(nn.Module):
    """One scalar-field GeoMPNN. Four copies are trained per experiment repetition."""
    def __init__(
        self,
        cfg: Dict,
        field: str,
        *,
        knn_query_chunk_size: int = 4096,
    ):
        super().__init__()
        self.cfg = cfg
        self.field = field
        self.knn_query_chunk_size = int(knn_query_chunk_size)
        if self.knn_query_chunk_size <= 0:
            raise ValueError("knn_query_chunk_size must be positive")

        hidden = int(cfg["hidden_dim"])
        self.surface_layers = int(cfg["surface_layers"])
        self.s2v_layers = int(cfg["surf2vol_layers"])
        self.surf_radius = float(cfg["surface_radius"])
        self.s2v_k = int(cfg["surf2vol_k"])

        canonical_fields = set(cfg["inlet_canonicalization_fields"])
        self.use_inlet_frame = field in canonical_fields
        self.use_base_frame = True  # Inlet model extends, rather than replaces, the SpH features.

        n_basis = int(cfg["n_basis"])
        spacing_s = float(cfg["sine_spacing_s"])
        domain_l = float(cfg["sine_domain_size_l"])
        self.dist_fn = EmbDist(n_basis, spacing_s, domain_l)
        self.angle_base = EmbAngle(n_basis, rotate=False)
        self.coord_base = EmbCoords(n_basis, spacing_s, domain_l, rotate=False)
        self.angle_inlet = EmbAngle(n_basis, rotate=True)
        self.coord_inlet = EmbCoords(n_basis, spacing_s, domain_l, rotate=True)

        num_frames = 1 + int(self.use_inlet_frame)
        # Paper feature accounting:
        # SpH = base + trail + sph + sine = 251;
        # Inlet adds 198 -> 449.
        self.node_dim = (
            2  # inlet velocity
            + 2 * num_frames  # normal in base frame (+ inlet frame for p/nu_t)
            + 3 * self.dist_fn.dim  # d(x), ||x||, ||x_tau|| plus sine embeddings
            + 2 * self.angle_base.dim * num_frames  # leading + trailing SpH
            + 2 * self.coord_base.dim * num_frames  # leading + trailing Cartesian + PE
        )
        self.edge_dim = self.dist_fn.dim + num_frames * (self.coord_base.dim + self.angle_base.dim)

        self.surface_radius_graph = ManualRadiusGraph(
            self.surf_radius, int(cfg["surface_max_neighbors"])
        )

        self.node_embedding_volume = MLP(self.node_dim, hidden, hidden, None)
        self.node_embedding_surface = MLP(self.node_dim, hidden, hidden, None)
        self.edge_embedding_surface = MLP(self.edge_dim, hidden, hidden, None)
        self.edge_embedding_s2v = MLP(self.edge_dim, hidden, hidden, None)

        mlp_norm = {
            "mlp_post": "post",
            "mlp_pre": "pre",
            "mlp_preAct": "preAct",
            "mlp_postAct": "postAct",
        }.get(cfg["norm_type"])
        if mlp_norm is None:
            raise ValueError(f"Unsupported norm_type={cfg['norm_type']!r}")

        self.surface_processors = nn.ModuleList([
            MessageProcessor(hidden, mlp_norm=mlp_norm, edge_norm=bool(cfg["edge_norm"]))
            for _ in range(self.surface_layers)
        ])
        self.s2v_processors = nn.ModuleList([
            MessageProcessor(hidden, mlp_norm=mlp_norm, edge_norm=bool(cfg["edge_norm"]))
            for _ in range(self.s2v_layers)
        ])

        # Paper Sec. 4.2: decoder is applied to final latent z(x), not [z(x), z_in(x)].
        dec = list(cfg["decoder_hidden"])
        self.decoder = nn.Sequential(
            nn.Linear(hidden, int(dec[0])), nn.ReLU(),
            nn.Linear(int(dec[0]), int(dec[1])), nn.ReLU(),
            nn.Linear(int(dec[1]), int(cfg["target_dim"])),
        )

    def _node_features(self, g) -> torch.Tensor:
        basis = g.inlet_vel[0]
        feats = [g.x, g.surface_normal]
        if self.use_inlet_frame:
            feats.append(change_basis(g.surface_normal, basis))

        # Distances: d(x), ||x||, ||x_tau||. Order is immaterial to dimensions but
        # kept explicit to mirror the paper's named features.
        feats.extend([
            self.dist_fn(g.dist[CLOSEST]),
            self.dist_fn(g.dist[NOSE]),
            self.dist_fn(g.dist[TAIL]),
        ])

        for key in (NOSE, TAIL):
            feats.append(self.angle_base(g.coords[key]))
            if self.use_inlet_frame:
                feats.append(self.angle_inlet(g.coords[key], basis))

        for key in (NOSE, TAIL):
            feats.append(self.coord_base(g.coords[key]))
            if self.use_inlet_frame:
                feats.append(self.coord_inlet(g.coords[key], basis))

        out = torch.cat(feats, dim=-1)
        if out.shape[-1] != self.node_dim:
            raise RuntimeError(f"node feature dim {out.shape[-1]} != expected {self.node_dim}")
        return out

    def _edge_features(self, displacement: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        dist = displacement.norm(dim=1, keepdim=True)
        feats = [self.dist_fn(dist), self.coord_base(displacement), self.angle_base(displacement)]
        if self.use_inlet_frame:
            feats.extend([self.coord_inlet(displacement, basis), self.angle_inlet(displacement, basis)])
        out = torch.cat(feats, dim=-1)
        if out.shape[-1] != self.edge_dim:
            raise RuntimeError(f"edge feature dim {out.shape[-1]} != expected {self.edge_dim}")
        return out

    @torch.no_grad()
    def _s2v_edges(self, surface_pos: torch.Tensor, volume_pos: torch.Tensor) -> torch.Tensor:
        """Build the exact paper kNN graph without a full N_volume x N_surface matrix."""
        if len(surface_pos) < self.s2v_k:
            raise ValueError(f"surface graph has {len(surface_pos)} nodes < k={self.s2v_k}")

        source_chunks = []
        target_chunks = []
        for start in range(0, len(volume_pos), self.knn_query_chunk_size):
            stop = min(start + self.knn_query_chunk_size, len(volume_pos))
            query = volume_pos[start:stop]
            distances = (surface_pos.unsqueeze(0) - query.unsqueeze(1)).norm(dim=-1)
            source_chunks.append(
                distances.topk(k=self.s2v_k, dim=1, largest=False).indices.flatten()
            )
            target_chunks.append(
                torch.arange(start, stop, device=volume_pos.device).repeat_interleave(self.s2v_k)
            )
        return torch.stack(
            [torch.cat(source_chunks), torch.cat(target_chunks)],
            dim=0,
        )

    def forward(self, graphs: Dict[str, object]) -> torch.Tensor:
        volume, surface = graphs[VOLUME], graphs[SURFACE]
        basis = volume.inlet_vel[0]

        raw_vol = self._node_features(volume)
        raw_surf = self._node_features(surface)
        x_vol = self.node_embedding_volume(raw_vol)
        x_surf = self.node_embedding_surface(raw_surf)

        # Paper Sec. 4.2: full surface graph, radius 0.05, max M=8; input edge
        # feature is [y-x, ||y-x||] plus the later Sine/SpH augmentations.
        surf_edge_index = self.surface_radius_graph(surface.pos)
        s_src, s_tgt = surf_edge_index
        surf_disp = surface.pos[s_src] - surface.pos[s_tgt]  # NO /surface_radius
        surf_e = self.edge_embedding_surface(self._edge_features(surf_disp, basis))
        for proc in self.surface_processors:
            x_surf, surf_e = proc(x_surf, x_surf, surf_edge_index, surf_e)

        # Directed kNN surface -> volume graph with k=8.
        s2v_edge_index = self._s2v_edges(surface.pos, volume.pos)
        sv_src, sv_tgt = s2v_edge_index
        s2v_disp = surface.pos[sv_src] - volume.pos[sv_tgt]
        s2v_e = self.edge_embedding_s2v(self._edge_features(s2v_disp, basis))

        # Paper's Surf2Vol update sequence only: edge residual update, mean aggregation,
        # node residual update. No extra volume-volume or independent-volume processor.
        for proc in self.s2v_processors:
            x_vol, s2v_e = proc(x_surf, x_vol, s2v_edge_index, s2v_e)

        return self.decoder(x_vol)


def build_four_models(
    cfg: Dict,
    *,
    knn_query_chunk_size: int = 4096,
) -> nn.ModuleDict:
    fields = ["x-velocity", "y-velocity", "pressure", "turbulent_viscosity"]
    return nn.ModuleDict(
        {
            field: GeoMPNN(
                cfg,
                field,
                knn_query_chunk_size=knn_query_chunk_size,
            )
            for field in fields
        }
    )
