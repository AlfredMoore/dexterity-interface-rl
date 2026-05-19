"""Depth feature extractor with FiLM-conditioned FK proprioception.

Architecture: ResNet backbone with per-stage FiLM conditioning from FK ->
              1x1 conv keypoint projection -> Spatial Softmax ->
              MLP head (with FK concatenation) -> output_dim regression.

FiLM (Feature-wise Linear Modulation, Perez et al. AAAI 2018) applies a
learned channel-wise affine transform  gamma * h + beta  to CNN feature
maps, where (gamma, beta) are generated from FK proprioception. This lets
the hand's joint configuration dynamically gate visual channels — e.g.
suppressing features in occluded regions and amplifying visible object cues.

FK conditioning signal:
    left_hand_base_pos (3) + right_hand_base_pos (3) +
    left_fingertip_pos (N_fingers * 3) + right_fingertip_pos (N_fingers * 3)

Two fusion paths ensure FK information reaches both early (spatial) and
late (semantic) stages:
    1. FiLM: modulates ResNet feature maps per-stage (spatial gating)
    2. Late concat: FK embedding concatenated with keypoints before MLP head
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# Extract the step counter from a 'depth_net_*_step_<N>.pth' filename. Matches
# both 'depth_net_step_28000.pth' and 'depth_net_epoch_005_step_10200.pth'.
# Returns None for files without a step tag (e.g. depth_net_best_val.pth) —
# those are ranked separately, see _load_latest_checkpoint.
_STEP_PATTERN = re.compile(r"step_(\d+)\.pth$")


def _step_of_checkpoint(path: str) -> int | None:
    m = _STEP_PATTERN.search(os.path.basename(path))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# FiLM layer & generator
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Apply channel-wise affine modulation: gamma * x + beta.

    gamma is initialized near 1 and beta near 0 so the layer starts
    close to an identity transform, preserving the pretrained / initial
    CNN features at the beginning of training.
    """

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W) feature map from a ResNet stage.
            gamma: (B, C) per-channel scale.
            beta:  (B, C) per-channel shift.
        """
        # reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class FiLMGenerator(nn.Module):
    """Generate (gamma, beta) pairs for each ResNet stage from FK input.

    Architecture:
        FK (fk_dim) -> shared MLP trunk -> per-stage linear heads -> (gamma_i, beta_i)

    gamma is parameterized as 1 + delta_gamma so the initial modulation
    is close to identity (scale=1, shift=0).
    """

    def __init__(self, fk_dim: int, stage_channels: list[int], hidden_dim: int = 128):
        """
        Args:
            fk_dim:         dimensionality of the FK conditioning vector.
            stage_channels: list of output channel counts for each ResNet stage,
                            e.g. [64, 128, 256, 512] for ResNet-10 with base=64.
            hidden_dim:     width of the shared trunk MLP.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(fk_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # one head per stage, outputs (gamma_delta, beta) concatenated
        self.heads = nn.ModuleList()
        for ch in stage_channels:
            head = nn.Linear(hidden_dim, ch * 2)
            # init: gamma_delta ≈ 0 (so gamma ≈ 1), beta ≈ 0
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.heads.append(head)

        self.stage_channels = stage_channels

    def forward(self, fk: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            fk: (B, fk_dim) FK proprioception vector.

        Returns:
            List of (gamma, beta) tuples, one per stage.
            gamma: (B, C_i), beta: (B, C_i).
        """
        h = self.trunk(fk)
        params = []
        for head, ch in zip(self.heads, self.stage_channels):
            out = head(h)                           # (B, 2 * C_i)
            gamma_delta, beta = out.split(ch, dim=-1)
            gamma = 1.0 + gamma_delta               # near-identity init
            params.append((gamma, beta))
        return params


# ---------------------------------------------------------------------------
# ResNet backbone (unchanged BasicBlock, FiLM applied between stages)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class ResNetFiLM(nn.Module):
    """ResNet with per-stage FiLM conditioning.

    Forward signature changes: forward(x, film_params) where film_params
    is a list of (gamma, beta) tuples from FiLMGenerator.

    FiLM is applied AFTER each stage's output (post residual + ReLU),
    before the feature map enters the next stage. This is the least
    invasive insertion point and empirically effective (TacFiLM, 2025).
    """

    def __init__(self, layers: list[int], in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.stage_channels: list[int] = []

        in_ch = base_channels
        for i, num_blocks in enumerate(layers):
            out_ch = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
            for _ in range(num_blocks - 1):
                blocks.append(BasicBlock(out_ch, out_ch, stride=1))
            self.stages.append(nn.Sequential(*blocks))
            self.film_layers.append(FiLMLayer())
            self.stage_channels.append(out_ch)
            in_ch = out_ch

        self.out_channels = in_ch

    def forward(
        self,
        x: torch.Tensor,
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if film_params is not None:
                gamma, beta = film_params[i]
                x = self.film_layers[i](x, gamma, beta)
        return x


def resnet10_film(in_channels: int = 1) -> ResNetFiLM:
    return ResNetFiLM(layers=[1, 1, 1, 1], in_channels=in_channels)


def resnet8_film(in_channels: int = 1) -> ResNetFiLM:
    return ResNetFiLM(layers=[1, 1, 1], in_channels=in_channels)


def resnet18_film(in_channels: int = 1) -> ResNetFiLM:
    return ResNetFiLM(layers=[2, 2, 2, 2], in_channels=in_channels)


# ---------------------------------------------------------------------------
# Spatial Softmax (unchanged)
# ---------------------------------------------------------------------------

class SpatialSoftmax(nn.Module):
    """Spatial softmax pooling.

    Returns expected (x, y) coordinate per channel, normalized to [-1, 1].
    Reference: Levine et al., "End-to-End Training of Deep Visuomotor
    Policies", 2016.
    """

    def __init__(self, temperature: float = 1.0, learnable_temperature: bool = False):
        super().__init__()
        t = torch.tensor(float(temperature))
        if learnable_temperature:
            self.temperature = nn.Parameter(t)
        else:
            self.register_buffer("temperature", t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        x_flat = x_grid.reshape(1, 1, -1)
        y_flat = y_grid.reshape(1, 1, -1)

        logits = x.reshape(B, C, -1) / self.temperature
        attn = F.softmax(logits, dim=-1)
        kx = (attn * x_flat).sum(dim=-1)
        ky = (attn * y_flat).sum(dim=-1)
        return torch.stack([kx, ky], dim=-1)


# ---------------------------------------------------------------------------
# Main network: DepthFeatureNet with FiLM
# ---------------------------------------------------------------------------

class DepthFeatureNetFiLM(nn.Module):
    """Depth + FK -> ResNet(FiLM) -> Spatial Softmax -> MLP -> output.

    Two FK fusion paths:
        1. FiLM: FK generates (gamma, beta) to modulate each ResNet stage.
           This gives FK spatial influence over early/mid feature extraction,
           enabling occlusion-aware channel gating.
        2. Late concat: FK embedding is concatenated with spatial-softmax
           keypoints before the MLP head, providing the head with explicit
           hand-state context for final regression.

    Args:
        output_dim:    regression target dimensionality.
        fk_dim:        FK conditioning vector size.
                       Example: 2 * (3 base_pos + N_fingers * 3 fingertip_pos).
                       For 4-finger Allegro: 2*(3+12) = 30.
                       For 5-finger: 2*(3+15) = 36.
        in_channels:   depth image channels (1 for raw depth).
        backbone:      "resnet10" | "resnet8" | "resnet18".
        num_keypoints: channels after 1x1 conv (soft keypoints).
        mlp_hidden:    hidden width of the MLP head.
        film_hidden:   hidden width of the FiLM generator trunk.
        fk_embed_dim:  FK embedding dim for late concatenation (0 to disable).
    """

    BACKBONES = {
        "resnet10": resnet10_film,
        "resnet8": resnet8_film,
        "resnet18": resnet18_film,
    }

    def __init__(
        self,
        output_dim: int,
        fk_dim: int = 24,
        in_channels: int = 1,
        backbone: str = "resnet10",
        num_keypoints: int = 64,
        mlp_hidden: int = 256,
        film_hidden: int = 128,
        fk_embed_dim: int = 32,
        spatial_softmax_temperature: float = 1.0,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        if backbone not in self.BACKBONES:
            raise ValueError(f"backbone must be one of {list(self.BACKBONES.keys())}, got {backbone}")

        # --- vision pathway ---
        self.backbone = self.BACKBONES[backbone](in_channels=in_channels)
        self.kp_conv = nn.Conv2d(self.backbone.out_channels, num_keypoints, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax(
            temperature=spatial_softmax_temperature,
            learnable_temperature=learnable_temperature,
        )

        # --- FiLM conditioning (path 1: spatial modulation) ---
        self.film_gen = FiLMGenerator(
            fk_dim=fk_dim,
            stage_channels=self.backbone.stage_channels,
            hidden_dim=film_hidden,
        )

        # --- late FK embedding (path 2: semantic concat) ---
        self.fk_embed_dim = fk_embed_dim
        if fk_embed_dim > 0:
            self.fk_embed = nn.Sequential(
                nn.Linear(fk_dim, fk_embed_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.fk_embed = None

        # --- MLP head: keypoints (+ optional FK embed) -> output ---
        head_input_dim = num_keypoints * 2 + fk_embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(
        self, depth: torch.Tensor, fk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            depth: (B, in_channels, H, W) preprocessed depth tensor.
            fk:    (B, fk_dim) FK proprioception vector.

        Returns:
            feature: (B, head_input_dim) keypoints [+ FK embed], before MLP.
            pred:    (B, output_dim) regression prediction.
        """
        # FiLM parameters from FK
        film_params = self.film_gen(fk)

        # vision pathway with FiLM conditioning
        feat_map = self.backbone(depth, film_params=film_params)
        kp_map = self.kp_conv(feat_map)
        keypoints = self.spatial_softmax(kp_map)
        kp_flat = keypoints.flatten(1)                     # (B, num_keypoints * 2)

        # late concatenation of FK embedding
        if self.fk_embed is not None:
            fk_emb = self.fk_embed(fk)                     # (B, fk_embed_dim)
            feature = torch.cat([kp_flat, fk_emb], dim=-1) # (B, num_kp*2 + fk_embed_dim)
        else:
            feature = kp_flat

        pred = self.head(feature)
        return feature, pred


# ---------------------------------------------------------------------------
# Backward-compatible wrapper: no-FiLM version (original API preserved)
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """Original ResNet without FiLM (kept for backward compatibility)."""

    def __init__(self, layers: list[int], in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        stages = []
        in_ch = base_channels
        for i, num_blocks in enumerate(layers):
            out_ch = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
            for _ in range(num_blocks - 1):
                blocks.append(BasicBlock(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))


def resnet10(in_channels: int = 1) -> ResNet:
    return ResNet(layers=[1, 1, 1, 1], in_channels=in_channels)


def resnet8(in_channels: int = 1) -> ResNet:
    return ResNet(layers=[1, 1, 1], in_channels=in_channels)


def resnet18(in_channels: int = 1) -> ResNet:
    return ResNet(layers=[2, 2, 2, 2], in_channels=in_channels)


class DepthFeatureNet(nn.Module):
    """Original DepthFeatureNet without FiLM (backward compatible)."""

    BACKBONES = {"resnet10": resnet10, "resnet8": resnet8, "resnet18": resnet18}

    def __init__(
        self,
        output_dim: int,
        in_channels: int = 1,
        backbone: str = "resnet10",
        num_keypoints: int = 64,
        mlp_hidden: int = 256,
        spatial_softmax_temperature: float = 1.0,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        self.backbone = self.BACKBONES[backbone](in_channels=in_channels)
        self.kp_conv = nn.Conv2d(self.backbone.out_channels, num_keypoints, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax(
            temperature=spatial_softmax_temperature,
            learnable_temperature=learnable_temperature,
        )
        self.head = nn.Sequential(
            nn.Linear(num_keypoints * 2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.backbone(depth)
        kp_map = self.kp_conv(feat_map)
        keypoints = self.spatial_softmax(kp_map)
        feature = keypoints.flatten(1)
        pred = self.head(feature)
        return feature, pred


# ---------------------------------------------------------------------------
# Wrapper: env-driven on-policy training (updated for FiLM)
# ---------------------------------------------------------------------------

# Hard-coded jar geometry bounds (metres) — matches the asset config in
# assets/open_the_jar/config/jar_config.yaml. Order is aligned with
# `bottleGeomCfg` / `jar_geom_keys`:
#     [body_radius, body_height, cap_radius, cap_height]
# Used by _preprocess_geom to clip + min-max normalize the geom slice of the
# regression target into [0, 1] before MSE, so the geom signal isn't drowned
# out by the larger-magnitude position errors.
_GEOM_MIN: tuple[float, float, float, float] = (0.03, 0.09, 0.02, 0.015)
_GEOM_MAX: tuple[float, float, float, float] = (0.05, 0.20, 0.05, 0.04)
# Index range of the geom block inside the target/pred vector.
# output_dim == 10 layout: body_pos(3) + cap_pos(3) + geom(4).
_GEOM_SLICE: slice = slice(6, 10)

@dataclass
class DepthFeatureExtractorCfg:
    """Configuration for the DepthFeatureExtractor wrapper.

    Set `use_film=True` and provide `fk_dim` to enable FiLM conditioning.
    When `use_film=False`, falls back to the original DepthFeatureNet.
    """

    enable: bool = False
    train: bool = True
    load_checkpoint: bool = False
    backbone: str = "resnet10"
    num_keypoints: int = 64
    mlp_hidden: int = 256
    lr: float = 1e-4
    loss_scale: float = 100.0
    checkpoint_save_interval: int = 10000
    near: float = 0.1
    far: float = 1.1
    target_components: list[str] = field(default_factory=list)
    log_dir: str | None = None
    # Optional explicit checkpoint path. When set (non-None / non-empty), it
    # takes precedence over the latest-pick logic that scans log_dir — use it
    # to pin deployment to a specific .pth file regardless of what else lives
    # in the same directory. Default None means "use the latest-pick logic".
    checkpoint_path: str | None = None

    # --- FiLM-specific ---
    use_film: bool = False
    fk_dim: int = 24                # 2*(3 base + 9 fingertips) for 3-finger hands
    film_hidden: int = 128          # FiLM generator trunk width
    fk_embed_dim: int = 32          # FK embedding for late concat (0 to disable)


class DepthFeatureExtractor:
    """Trainable wrapper, with optional FiLM conditioning from FK.

    When cfg.use_film is True:
        - Instantiates DepthFeatureNetFiLM
        - step() requires an additional `fk` argument
        - The FK vector should be assembled by the env as:
          [left_base_pos(3), right_base_pos(3),
           left_fingertips(N*3), right_fingertips(N*3)]

    When cfg.use_film is False:
        - Falls back to original DepthFeatureNet (full backward compat)
        - step() ignores `fk` if provided
    """

    def __init__(self, cfg: DepthFeatureExtractorCfg, output_dim: int, device: torch.device | str):
        self.cfg = cfg
        self.device = device
        self.output_dim = output_dim
        self.step_count: int = 0

        if cfg.use_film:
            self.net = DepthFeatureNetFiLM(
                output_dim=output_dim,
                fk_dim=cfg.fk_dim,
                in_channels=1,
                backbone=cfg.backbone,
                num_keypoints=cfg.num_keypoints,
                mlp_hidden=cfg.mlp_hidden,
                film_hidden=cfg.film_hidden,
                fk_embed_dim=cfg.fk_embed_dim,
            ).to(self.device)
        else:
            self.net = DepthFeatureNet(
                output_dim=output_dim,
                in_channels=1,
                backbone=cfg.backbone,
                num_keypoints=cfg.num_keypoints,
                mlp_hidden=cfg.mlp_hidden,
            ).to(self.device)

        # Cache geom bounds as device tensors for _preprocess_geom (shape (4,)).
        self._geom_min = torch.tensor(_GEOM_MIN, device=device, dtype=torch.float32)
        self._geom_max = torch.tensor(_GEOM_MAX, device=device, dtype=torch.float32)
        self._geom_slice = _GEOM_SLICE

        self.log_dir = cfg.log_dir or os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs_depth")
        os.makedirs(self.log_dir, exist_ok=True)

        if cfg.load_checkpoint:
            if cfg.checkpoint_path:
                self._load_specific_checkpoint(cfg.checkpoint_path)
            else:
                self._load_latest_checkpoint()

        if cfg.train:
            self.optimizer: torch.optim.Optimizer | None = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
            self.net.train()
        else:
            self.optimizer = None
            self.net.eval()

    def _preprocess_depth(self, raw_depth: torch.Tensor) -> torch.Tensor:
        """Normalize raw depth into (B, 1, H, W) in [0, 1]."""
        if raw_depth.dim() == 4 and raw_depth.shape[-1] == 1:
            depth = raw_depth.permute(0, 3, 1, 2)
        elif raw_depth.dim() == 3:
            depth = raw_depth.unsqueeze(1)
        else:
            depth = raw_depth
        depth = torch.nan_to_num(depth, nan=self.cfg.far, posinf=self.cfg.far, neginf=0.0)
        depth = depth.clamp(min=self.cfg.near, max=self.cfg.far)
        depth = (depth - self.cfg.near) / (self.cfg.far - self.cfg.near)
        return depth

    def _preprocess_geom(self, geom: torch.Tensor) -> torch.Tensor:
        """Clip + min-max normalize bottle geometry into [0, 1].

        Args:
            geom: (..., 4) tensor in metres, last dim ordered
                  [body_radius, body_height, cap_radius, cap_height].

        Returns:
            (..., 4) tensor in [0, 1] with the same leading shape.
        """
        g = geom.clamp(min=self._geom_min, max=self._geom_max)
        return (g - self._geom_min) / (self._geom_max - self._geom_min)

    def un_preprocess_geom(self, geom_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of _preprocess_geom: map [0, 1] back to metres."""
        return geom_norm * (self._geom_max - self._geom_min) + self._geom_min

    def _forward(self, depth: torch.Tensor, fk: torch.Tensor | None):
        """Dispatch to FiLM or vanilla forward."""
        if self.cfg.use_film and isinstance(self.net, DepthFeatureNetFiLM):
            assert fk is not None, "FiLM mode requires FK input"
            return self.net(depth, fk)
        else:
            return self.net(depth)

    def step(
        self,
        raw_depth: torch.Tensor,
        target: torch.Tensor,
        fk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One forward (+ optional backward) pass.

        Args:
            raw_depth: (B, H, W, 1) or (B, 1, H, W) depth from TiledCamera.
            target:    (B, output_dim) ground-truth target from env (metres).
                       The geom slab `target[..., 6:10]` is internally
                       normalized to [0, 1] before loss; the original tensor
                       is not mutated (we clone first).
            fk:        (B, fk_dim) FK proprioception. Required when use_film=True.

        Returns:
            feature: (B, feature_dim) detached.
            pred:    (B, output_dim) detached. Convention:
                       pred[..., :6] is in metres (positions);
                       pred[..., 6:10] is in normalized [0, 1] space —
                       use `un_preprocess_geom` to map back to metres.
            loss:    scalar (detached) if training; None in eval mode.
        """
        if self.cfg.train and self.optimizer is not None:
            with torch.enable_grad(), torch.inference_mode(False):
                depth_in = self._preprocess_depth(raw_depth.clone())
                target_in = target.clone()
                # Normalize geom slab so its MSE contribution is on the same
                # order as the (metre-scale) position slabs.
                target_in[..., self._geom_slice] = self._preprocess_geom(
                    target_in[..., self._geom_slice]
                )
                fk_in = fk.clone() if fk is not None else None

                self.optimizer.zero_grad(set_to_none=True)
                feature, pred = self._forward(depth_in, fk_in)
                loss = F.mse_loss(pred, target_in) * self.cfg.loss_scale
                loss.backward()
                self.optimizer.step()

                self.step_count += 1
                if (
                    self.cfg.checkpoint_save_interval > 0
                    and self.step_count % self.cfg.checkpoint_save_interval == 0
                ):
                    self.save_checkpoint()

                return feature.detach(), pred.detach(), loss.detach()
        else:
            with torch.no_grad():
                depth_in = self._preprocess_depth(raw_depth.clone())
                fk_in = fk.clone() if fk is not None else None
                feature, pred = self._forward(depth_in, fk_in)
            return feature, pred, None

    def save_checkpoint(self, tag: str | None = None):
        tag = tag if tag is not None else f"step_{self.step_count}"
        path = os.path.join(self.log_dir, f"depth_net_{tag}.pth")
        torch.save(self.net.state_dict(), path)
        return path

    def _load_latest_checkpoint(self):
        # Inference vs. training need different "which ckpt is best" semantics:
        #   * inference (cfg.train=False) -> the val-best snapshot, because the
        #     largest-step file is just whichever epoch happened to finish last
        #     and is typically worse than best_val if the run overfit.
        #   * training (cfg.train=True)   -> the largest-step file, so a resume
        #     continues from where the run left off rather than rewinding to
        #     the val-best snapshot.
        # Ranking by step counter embedded in the filename instead of ctime —
        # ctime is unreliable after a cp/rsync (all files end up timestamped
        # at the copy moment). best_val.pth carries no step tag and is handled
        # separately.
        best_val = os.path.join(self.log_dir, "depth_net_best_val.pth")
        if not self.cfg.train and os.path.exists(best_val):
            latest = best_val
        else:
            files = glob.glob(os.path.join(self.log_dir, "depth_net_*.pth"))
            step_files = [(p, _step_of_checkpoint(p)) for p in files]
            step_files = [(p, s) for p, s in step_files if s is not None]
            if not step_files:
                msg = (
                    f"[DepthFeatureExtractor] no step-tagged checkpoint found "
                    f"under {self.log_dir}"
                )
                # In inference mode a missing checkpoint means the network is
                # silently random-initialised — every downstream consumer then
                # gets garbage predictions with no obvious failure signal. Hard
                # fail. Training keeps the old warn-and-continue path so a
                # fresh run can start from scratch.
                if not self.cfg.train:
                    raise FileNotFoundError(msg)
                print(msg)
                return
            latest = max(step_files, key=lambda ps: ps[1])[0]
        print(f"[DepthFeatureExtractor] loading checkpoint: {latest}")
        self.net.load_state_dict(torch.load(latest, map_location=self.device, weights_only=True))

    def _load_specific_checkpoint(self, path: str):
        """Load a single checkpoint by path, bypassing the latest-pick logic.

        Used when cfg.checkpoint_path is set — overrides the directory-scan
        path entirely. Missing file raises FileNotFoundError in both train and
        inference modes since an explicitly-named path that doesn't resolve is
        a configuration error either way.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[DepthFeatureExtractor] checkpoint not found at {path}"
            )
        print(f"[DepthFeatureExtractor] loading specified checkpoint: {path}")
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, W = 4, 120, 160

    # --- FiLM version ---
    fk_dim = 24   # 2 * (3 base + 3*3 fingertips)
    output_dim = 10  # body_pos(3) + cap_pos(3) + size(4)

    net = DepthFeatureNetFiLM(
        output_dim=output_dim,
        fk_dim=fk_dim,
        backbone="resnet10",
        num_keypoints=64,
        fk_embed_dim=32,
    ).to(device)

    depth = torch.randn(B, 1, H, W, device=device)
    fk = torch.randn(B, fk_dim, device=device)

    feat, pred = net(depth, fk)
    print(f"[FiLM]  feature: {feat.shape}, pred: {pred.shape}")
    # feature: (4, 160)  = 64*2 + 32
    # pred:    (4, 10)

    # --- No-FiLM version (backward compat) ---
    net_vanilla = DepthFeatureNet(
        output_dim=output_dim,
        backbone="resnet10",
        num_keypoints=64,
    ).to(device)

    feat_v, pred_v = net_vanilla(depth)
    print(f"[Vanilla] feature: {feat_v.shape}, pred: {pred_v.shape}")

    # --- Wrapper test ---
    cfg = DepthFeatureExtractorCfg(
        enable=True,
        train=True,
        use_film=True,
        fk_dim=fk_dim,
        fk_embed_dim=32,
        log_dir="/tmp/depth_test",
    )
    extractor = DepthFeatureExtractor(cfg, output_dim=output_dim, device=device)

    raw_depth = torch.randn(B, H, W, 1, device=device)  # simulating TiledCamera output
    target = torch.randn(B, output_dim, device=device)
    fk_input = torch.randn(B, fk_dim, device=device)

    feat_out, pred_out, loss = extractor.step(raw_depth, target, fk=fk_input)
    print(f"[Wrapper] feature: {feat_out.shape}, pred: {pred_out.shape}, loss: {loss.item():.4f}")

    # parameter count comparison
    n_film = sum(p.numel() for p in net.parameters())
    n_vanilla = sum(p.numel() for p in net_vanilla.parameters())
    print(f"\nParam count — FiLM: {n_film:,}  Vanilla: {n_vanilla:,}  "
          f"Overhead: {n_film - n_vanilla:,} ({(n_film - n_vanilla) / n_vanilla * 100:.1f}%)")