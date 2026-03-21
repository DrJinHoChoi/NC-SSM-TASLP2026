#!/usr/bin/env python3
# coding=utf-8
# NC-SSM Vision: Task Heads for Lane Detection & Critical Object Spotting
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.
# Target: ICCV 2027 -- Tunnel / Rapid Illumination Change Environments
"""
NC-SSM Vision Task Heads -- Lane Detection & Critical Object Spotting
======================================================================

Audio KWS -> Vision analog:

  Audio (Interspeech 2026)                Vision (ICCV 2027)
  ----------------------                  -------------------
  Keyword Spotting (KWS)                  Critical Object Spotting (COS)
    "Hey Siri" detection                    Stop sign / Pedestrian detection
    12-class classification                 N-class detection + localization
    Always-on, ultra-low power              Always-on ADAS, ultra-low power
    Factory noise robustness                Tunnel/night robustness

  Task structure:
    KWS = "is keyword X present?"           COS = "is critical object X at (x,y,w,h)?"
    Output: class logits (12)               Output: class + bbox (N * 5)
    Fixed-size input (1s audio)             Fixed-size input (crop/resize image)

Two task heads sharing the same NC-SSM Vision backbone:

  Task 1: Lane Detection (CULane benchmark)
    - Row-anchor based (Ultra-Fast Lane style)
    - Output: 4 lanes x H_anchors x (W_grids + 1)
    - Lightest possible detection task

  Task 2: Critical Object Spotting (ExDark / ACDC)
    - Anchor-free single-stage detector
    - Focus: pedestrian, vehicle, traffic sign in adverse conditions
    - "Visual Keyword Spotting" -- detect critical objects, not everything

Both heads are plug-and-play on NanoMamba Vision backbone (ncssm_vision.py).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import backbone from ncssm_vision.py
from ncssm_vision import (
    NanoMambaVision,
    NCSSMVision,
    NanoMambaVisionBlock,
    VisibilityEstimator,
    DualNormMoE,
    SpatialRetinexBypass,
)


# ============================================================================
# Task 1: Lane Detection Head (Row-Anchor Style)
# ============================================================================

class LaneHead(nn.Module):
    """Ultra-lightweight lane detection head.

    Inspired by Ultra-Fast-Lane-Detection (UFLD):
      - Formulates lane detection as row-wise classification
      - Each row anchor selects one of W grid cells (or "no lane")
      - Output: n_lanes x n_anchors x (n_grids + 1)

    Why row-anchor (not segmentation)?
      - Segmentation: output H*W mask -> heavy
      - Row-anchor: output n_lanes * n_anchors * n_grids -> ultra-light
      - CULane standard: 18 row anchors, 100 grid cells -> 7,272 outputs
      - vs segmentation: 590 * 1640 = 967,600 outputs (133x more!)

    Audio analog:
      KWS classifier: Linear(d_model, 12) -> 12 keyword logits
      Lane classifier: Linear(d_model, n_lanes * n_anchors * (n_grids+1))

    Params: d_model * n_lanes * n_anchors * (n_grids + 1) + bias
    """

    def __init__(self, d_model, n_lanes=4, n_anchors=18, n_grids=100):
        super().__init__()
        self.n_lanes = n_lanes
        self.n_anchors = n_anchors
        self.n_grids = n_grids

        # Row-wise spatial pooling + per-anchor prediction
        # Instead of GAP (loses all spatial info), pool to n_anchors rows
        # then predict per-row lane positions
        d_mid = max(d_model * 2, 32)
        self.row_fc = nn.Linear(d_model, d_mid)
        self.lane_fc = nn.Linear(d_mid, n_lanes * 2)  # per-row: (x, conf) * n_lanes
        self.dropout = nn.Dropout(0.1)
        self._use_regression = True

    def forward(self, features):
        """
        Args:
            features: (B, N_patches, d_model) from NC-SSM backbone
        Returns:
            lane_preds: (B, n_lanes, n_anchors, 2) -- (x_position, confidence)
        """
        B, N, d = features.shape

        # Pool to n_anchors rows (preserves vertical spatial info)
        # (B, N, d) -> (B, d, N) -> adaptive_avg_pool -> (B, d, n_anchors) -> (B, n_anchors, d)
        x = features.transpose(1, 2)  # (B, d, N)
        x = F.adaptive_avg_pool1d(x, self.n_anchors)  # (B, d, n_anchors)
        x = x.transpose(1, 2)  # (B, n_anchors, d)

        # Per-row prediction
        x = F.silu(self.row_fc(x))  # (B, n_anchors, d_mid)
        x = self.dropout(x)
        preds = self.lane_fc(x)  # (B, n_anchors, n_lanes * 2)

        # Reshape to (B, n_lanes, n_anchors, 2)
        preds = preds.view(B, self.n_anchors, self.n_lanes, 2)
        preds = preds.permute(0, 2, 1, 3)  # (B, n_lanes, n_anchors, 2)
        return preds

    def decode(self, preds, img_w=1640, img_h=590, conf_threshold=0.5):
        """Decode regression predictions to lane coordinates.

        Args:
            preds: (B, n_lanes, n_anchors, 2) -- (x_pos, confidence)
            img_w: original image width
            img_h: original image height
        Returns:
            lanes: list of list of (n_valid, 2) tensors per batch
        """
        B = preds.size(0)
        x_pos = torch.sigmoid(preds[..., 0])   # normalized x in [0, 1]
        conf = torch.sigmoid(preds[..., 1])     # confidence in [0, 1]

        anchor_ys = torch.linspace(0.4, 1.0, self.n_anchors)

        results = []
        for b in range(B):
            batch_lanes = []
            for l in range(self.n_lanes):
                mask = conf[b, l] > conf_threshold
                if mask.sum() > 2:
                    xs = (x_pos[b, l, mask] * img_w).cpu()
                    ys = (anchor_ys[mask] * img_h).cpu()
                    lane = torch.stack([xs, ys], dim=1)
                    batch_lanes.append(lane)
            results.append(batch_lanes)
        return results


class LaneDetectionLoss(nn.Module):
    """Lane detection loss combining classification + structural priors.

    Components:
      1. Cross-entropy: row-anchor classification (main loss)
      2. Similarity loss: adjacent rows should predict similar x positions
         (lanes are smooth curves, not random jumps)
      3. Existence loss: binary CE for lane existence per anchor

    Audio analog:
      KWS loss = CrossEntropy(logits, keyword_label)
      Lane loss = CE(row_logits, grid_label) + smoothness + existence
    """

    def __init__(self, sim_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, gt_x, gt_conf):
        """
        Args:
            preds: (B, n_lanes, n_anchors, 2) -- predicted (x_pos, conf)
            gt_x: (B, n_lanes, n_anchors) -- ground truth x position [0,1]
            gt_conf: (B, n_lanes, n_anchors) -- ground truth lane existence {0,1}
        Returns:
            loss: scalar
        """
        pred_x = torch.sigmoid(preds[..., 0])
        pred_conf = preds[..., 1]

        # 1. Position loss (L1, only where lane exists)
        mask = gt_conf > 0.5
        n_pos = mask.sum().clamp(min=1)
        pos_loss = (F.l1_loss(pred_x, gt_x, reduction='none') * mask).sum() / n_pos

        # 2. Confidence loss (BCE)
        conf_loss = F.binary_cross_entropy_with_logits(pred_conf, gt_conf)

        # 3. Smoothness loss (adjacent anchors should be continuous)
        if pred_x.size(2) > 1:
            diff = (pred_x[:, :, 1:] - pred_x[:, :, :-1]).pow(2)
            sim_loss = (diff * mask[:, :, 1:]).sum() / mask[:, :, 1:].sum().clamp(1)
        else:
            sim_loss = torch.tensor(0.0, device=preds.device)

        return pos_loss + conf_loss + self.sim_weight * sim_loss


# ============================================================================
# Task 2: Critical Object Spotting Head (Anchor-Free)
# ============================================================================

class CriticalObjectHead(nn.Module):
    """Critical Object Spotting -- "Visual Keyword Spotting".

    Audio KWS: detect 10 keywords from 1s audio
    Visual COS: detect N critical objects from image

    Focus classes (safety-critical, tunnel/night relevant):
      - Pedestrian (most critical)
      - Vehicle (front/rear)
      - Traffic sign (stop, speed limit)
      - Traffic light
      - Construction cone/barrier

    Architecture: anchor-free, center-point based (CenterNet-style)
      - Predict heatmap: where are object centers?
      - Predict offset: sub-pixel center refinement
      - Predict size: (w, h) of bounding box
      - No NMS needed (like YOLOv10's NMS-free design)

    Why CenterNet-style (not YOLO)?
      - Simplest possible detection head
      - Heatmap = classification map (analogous to KWS frame-level posteriors)
      - No anchors, no NMS -> minimal post-processing
      - Output: 3 maps of (n_classes, H_out, W_out), (2, H_out, W_out), (2, H_out, W_out)

    Params: very small (3 conv layers)
    """

    def __init__(self, d_model, n_classes=5, n_patches_h=14, n_patches_w=14):
        super().__init__()
        self.n_classes = n_classes
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w

        # Heatmap head: (d_model) -> (n_classes) per spatial location
        # "Is there a [pedestrian/vehicle/sign] centered here?"
        # Directly analogous to KWS: "Is keyword [yes/no/stop] present?"
        self.heatmap = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, n_classes, 1, bias=True),
        )
        # Initialize bias for focal loss (rare positive samples)
        nn.init.constant_(self.heatmap[-1].bias, -2.19)  # log(0.01/0.99)

        # Offset head: sub-pixel center refinement (2 channels: dx, dy)
        self.offset = nn.Conv2d(d_model, 2, 1, bias=True)

        # Size head: bounding box width and height (2 channels: w, h)
        self.size = nn.Conv2d(d_model, 2, 1, bias=True)

    def forward(self, features):
        """
        Args:
            features: (B, N_patches, d_model) from NC-SSM backbone
        Returns:
            heatmap: (B, n_classes, H, W) center heatmaps
            offset: (B, 2, H, W) sub-pixel offset
            size: (B, 2, H, W) object size (w, h)
        """
        B, N, d = features.shape
        H, W = self.n_patches_h, self.n_patches_w

        # Reshape to spatial: (B, N, d) -> (B, d, H, W)
        x = features.transpose(1, 2).reshape(B, d, H, W)

        heatmap = self.heatmap(x)     # (B, n_classes, H, W)
        offset = self.offset(x)       # (B, 2, H, W)
        size = self.size(x)           # (B, 2, H, W)

        return heatmap, offset, size

    def decode(self, heatmap, offset, size, score_threshold=0.3,
               img_w=1640, img_h=590, top_k=20):
        """Decode predictions to bounding boxes.

        Args:
            heatmap: (B, n_classes, H, W)
            offset: (B, 2, H, W)
            size: (B, 2, H, W)
            score_threshold: minimum confidence
            img_w, img_h: original image size
        Returns:
            detections: list of (N_det, 6) tensors [x1, y1, x2, y2, score, class]
        """
        B, C, H, W = heatmap.shape

        # Apply sigmoid to get probabilities
        scores = heatmap.sigmoid()  # (B, C, H, W)

        # Simple pseudo-NMS: 3x3 max pooling
        scores_pool = F.max_pool2d(scores, 3, stride=1, padding=1)
        keep = (scores == scores_pool) & (scores > score_threshold)

        results = []
        for b in range(B):
            det_list = []
            for c in range(C):
                mask = keep[b, c]  # (H, W)
                if mask.sum() == 0:
                    continue
                ys, xs = torch.where(mask)
                sc = scores[b, c, ys, xs]

                # Top-k per class
                if len(sc) > top_k:
                    topk_idx = sc.topk(top_k).indices
                    ys, xs, sc = ys[topk_idx], xs[topk_idx], sc[topk_idx]

                # Center coordinates with sub-pixel offset
                cx = (xs.float() + offset[b, 0, ys, xs]) / W * img_w
                cy = (ys.float() + offset[b, 1, ys, xs]) / H * img_h

                # Bounding box
                w = size[b, 0, ys, xs].exp() / W * img_w
                h = size[b, 1, ys, xs].exp() / H * img_h

                x1 = (cx - w / 2).clamp(min=0)
                y1 = (cy - h / 2).clamp(min=0)
                x2 = (cx + w / 2).clamp(max=img_w)
                y2 = (cy + h / 2).clamp(max=img_h)

                cls_id = torch.full_like(sc, c)
                dets = torch.stack([x1, y1, x2, y2, sc, cls_id], dim=1)
                det_list.append(dets)

            if det_list:
                results.append(torch.cat(det_list, dim=0))
            else:
                results.append(torch.zeros(0, 6, device=heatmap.device))
        return results


class CriticalObjectLoss(nn.Module):
    """CenterNet-style loss for Critical Object Spotting.

    Components:
      1. Focal loss on heatmap (handles class imbalance, rare objects)
      2. L1 loss on offset (sub-pixel refinement)
      3. L1 loss on size (bounding box regression)

    Audio analog:
      KWS = CrossEntropy + LabelSmoothing
      COS = FocalLoss + L1 offset + L1 size
    """

    def __init__(self, alpha=2.0, beta=4.0, offset_weight=1.0, size_weight=0.1):
        super().__init__()
        self.alpha = alpha  # focal loss alpha
        self.beta = beta    # focal loss beta
        self.offset_weight = offset_weight
        self.size_weight = size_weight

    def focal_loss(self, pred, target):
        """Modified focal loss (CornerNet/CenterNet style).

        Args:
            pred: (B, C, H, W) sigmoid heatmap predictions
            target: (B, C, H, W) Gaussian heatmap targets
        """
        pred = pred.sigmoid().clamp(1e-6, 1 - 1e-6)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Positive: (1 - pred)^alpha * log(pred)
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        # Negative: (1 - target)^beta * pred^alpha * log(1 - pred)
        neg_loss = -(
            (1 - target) ** self.beta
            * (pred ** self.alpha)
            * torch.log(1 - pred)
            * neg_mask
        )

        n_pos = pos_mask.sum().clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / n_pos

    def forward(self, heatmap, offset, size,
                gt_heatmap, gt_offset, gt_size, gt_mask):
        """
        Args:
            heatmap: (B, C, H, W) predicted heatmaps
            offset: (B, 2, H, W) predicted offsets
            size: (B, 2, H, W) predicted sizes
            gt_heatmap: (B, C, H, W) ground truth Gaussian heatmaps
            gt_offset: (B, 2, H, W) ground truth offsets
            gt_size: (B, 2, H, W) ground truth sizes
            gt_mask: (B, 1, H, W) mask for positive locations
        """
        # 1. Focal loss on heatmap
        hm_loss = self.focal_loss(heatmap, gt_heatmap)

        # 2. L1 loss on offset (only at positive locations)
        n_pos = gt_mask.sum().clamp(min=1)
        off_loss = (F.l1_loss(offset, gt_offset, reduction='none') * gt_mask
                    ).sum() / n_pos

        # 3. L1 loss on size (only at positive locations)
        sz_loss = (F.l1_loss(size, gt_size, reduction='none') * gt_mask
                   ).sum() / n_pos

        return hm_loss + self.offset_weight * off_loss + self.size_weight * sz_loss


# ============================================================================
# Full Models: Backbone + Task Head
# ============================================================================

class NanoMambaLaneDetector(nn.Module):
    """NanoMamba Lane Detector: NC-SSM backbone + Row-Anchor Lane Head.

    End-to-end pipeline:
      Image -> [Retinex (0p)] -> PatchEmbed -> VisEst (0p)
      -> [DualNorm (1p)] -> NC-SSM x N -> LaneHead -> 4 lanes

    Audio analog pipeline:
      Audio -> [SS+Bypass (0p)] -> STFT -> SNR Est (0p)
      -> [DualPCEN (1p)] -> NC-SSM x N -> Classifier -> 12 keywords
    """

    def __init__(self, img_size=288, patch_size=16, n_lanes=4,
                 n_anchors=18, n_grids=100,
                 d_model=24, d_state=6, n_repeats=4, **kwargs):
        super().__init__()

        self.backbone = NanoMambaVision(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            n_classes=1,  # dummy, not used
            d_model=d_model, d_state=d_state, d_conv=3, expand=1.0,
            n_layers=1, n_repeats=n_repeats, weight_sharing=True,
            use_dual_norm=True, use_retinex=True, use_nc_ssm=True,
            **kwargs)
        # Remove classifier from backbone (we use our own head)
        self.backbone.classifier = nn.Identity()
        self.backbone.final_norm = nn.Identity()

        self.head = LaneHead(
            d_model=d_model, n_lanes=n_lanes,
            n_anchors=n_anchors, n_grids=n_grids)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image
        Returns:
            lane_logits: (B, n_lanes, n_anchors, n_grids+1)
        """
        # Backbone forward (minus final_norm and classifier)
        if self.backbone.retinex is not None:
            x = self.backbone.retinex(x)

        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        vis = self.backbone.vis_estimator(tokens)

        norm_gate = None
        if self.backbone.dual_norm is not None:
            tokens, norm_gate = self.backbone.dual_norm(tokens, None)

        if self.backbone.weight_sharing:
            for _ in range(self.backbone.n_repeats):
                tokens = self.backbone.shared_block(tokens, vis, norm_gate)
        else:
            for block in self.backbone.blocks:
                tokens = block(tokens, vis, norm_gate)

        return self.head(tokens)


class NanoMambaCriticalDetector(nn.Module):
    """NanoMamba Critical Object Detector: NC-SSM backbone + CenterNet Head.

    "Visual Keyword Spotting" -- detect safety-critical objects only.
    Not a general-purpose detector. Focused like KWS is focused.

    Critical classes (default 5):
      0: pedestrian
      1: vehicle
      2: traffic_sign
      3: traffic_light
      4: construction_barrier
    """

    CRITICAL_CLASSES = [
        'pedestrian', 'vehicle', 'traffic_sign',
        'traffic_light', 'construction_barrier'
    ]

    def __init__(self, img_size=288, patch_size=16, n_classes=5,
                 d_model=24, d_state=6, n_repeats=4, **kwargs):
        super().__init__()

        n_patches_h = img_size // patch_size
        n_patches_w = img_size // patch_size

        self.backbone = NanoMambaVision(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            n_classes=1,
            d_model=d_model, d_state=d_state, d_conv=3, expand=1.0,
            n_layers=1, n_repeats=n_repeats, weight_sharing=True,
            use_dual_norm=True, use_retinex=True, use_nc_ssm=True,
            **kwargs)
        self.backbone.classifier = nn.Identity()
        self.backbone.final_norm = nn.Identity()

        self.head = CriticalObjectHead(
            d_model=d_model, n_classes=n_classes,
            n_patches_h=n_patches_h, n_patches_w=n_patches_w)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image
        Returns:
            heatmap: (B, n_classes, H_out, W_out)
            offset: (B, 2, H_out, W_out)
            size: (B, 2, H_out, W_out)
        """
        if self.backbone.retinex is not None:
            x = self.backbone.retinex(x)

        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        vis = self.backbone.vis_estimator(tokens)

        norm_gate = None
        if self.backbone.dual_norm is not None:
            tokens, norm_gate = self.backbone.dual_norm(tokens, None)

        if self.backbone.weight_sharing:
            for _ in range(self.backbone.n_repeats):
                tokens = self.backbone.shared_block(tokens, vis, norm_gate)
        else:
            for block in self.backbone.blocks:
                tokens = block(tokens, vis, norm_gate)

        return self.head(tokens)


class NanoMambaMultiTaskDetector(nn.Module):
    """Multi-Task: Lane Detection + Critical Object Spotting (shared backbone).

    One backbone, two heads. Maximum parameter efficiency.
    Audio analog: one NanoMamba backbone, multiple wake words.

    Total params ~ backbone (25K) + lane head (~3K) + det head (~1K) = ~29K
    """

    def __init__(self, img_size=288, patch_size=16,
                 n_lanes=4, n_anchors=18, n_grids=100,
                 n_det_classes=5,
                 d_model=24, d_state=6, n_repeats=4, **kwargs):
        super().__init__()

        n_patches_h = img_size // patch_size
        n_patches_w = img_size // patch_size

        self.backbone = NanoMambaVision(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            n_classes=1,
            d_model=d_model, d_state=d_state, d_conv=3, expand=1.0,
            n_layers=1, n_repeats=n_repeats, weight_sharing=True,
            use_dual_norm=True, use_retinex=True, use_nc_ssm=True,
            **kwargs)
        self.backbone.classifier = nn.Identity()
        self.backbone.final_norm = nn.Identity()

        self.lane_head = LaneHead(
            d_model=d_model, n_lanes=n_lanes,
            n_anchors=n_anchors, n_grids=n_grids)

        self.det_head = CriticalObjectHead(
            d_model=d_model, n_classes=n_det_classes,
            n_patches_h=n_patches_h, n_patches_w=n_patches_w)

    def _backbone_forward(self, x):
        """Shared backbone forward pass."""
        if self.backbone.retinex is not None:
            x = self.backbone.retinex(x)

        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        vis = self.backbone.vis_estimator(tokens)

        norm_gate = None
        if self.backbone.dual_norm is not None:
            tokens, norm_gate = self.backbone.dual_norm(tokens, None)

        if self.backbone.weight_sharing:
            for _ in range(self.backbone.n_repeats):
                tokens = self.backbone.shared_block(tokens, vis, norm_gate)
        else:
            for block in self.backbone.blocks:
                tokens = block(tokens, vis, norm_gate)

        return tokens

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image
        Returns:
            dict with 'lanes' and 'detections'
        """
        tokens = self._backbone_forward(x)

        lanes = self.lane_head(tokens)
        heatmap, offset, size = self.det_head(tokens)

        return {
            'lanes': lanes,
            'heatmap': heatmap,
            'offset': offset,
            'size': size,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_lane_detector_nano(img_size=192, n_lanes=4, **kwargs):
    """Nano lane detector: ~15K params. patch_size=32 -> 36 patches."""
    return NanoMambaLaneDetector(
        img_size=img_size, patch_size=32, n_lanes=n_lanes,
        n_anchors=18, n_grids=50,
        d_model=16, d_state=4, n_repeats=4, **kwargs)


def create_lane_detector_tiny(img_size=192, n_lanes=4, **kwargs):
    """Tiny lane detector: ~25K params. patch_size=32 -> 36 patches."""
    return NanoMambaLaneDetector(
        img_size=img_size, patch_size=32, n_lanes=n_lanes,
        n_anchors=18, n_grids=100,
        d_model=24, d_state=6, n_repeats=4, **kwargs)


def create_critical_detector_nano(img_size=192, n_classes=5, **kwargs):
    """Nano critical object detector: ~15K params."""
    return NanoMambaCriticalDetector(
        img_size=img_size, patch_size=32, n_classes=n_classes,
        d_model=16, d_state=4, n_repeats=4, **kwargs)


def create_critical_detector_tiny(img_size=192, n_classes=5, **kwargs):
    """Tiny critical object detector: ~22K params."""
    return NanoMambaCriticalDetector(
        img_size=img_size, patch_size=32, n_classes=n_classes,
        d_model=24, d_state=6, n_repeats=4, **kwargs)


def create_multitask_detector_tiny(img_size=192, **kwargs):
    """Tiny multi-task (lane + detection): ~30K params."""
    return NanoMambaMultiTaskDetector(
        img_size=img_size, patch_size=32,
        n_lanes=4, n_anchors=18, n_grids=100,
        n_det_classes=5,
        d_model=24, d_state=6, n_repeats=4, **kwargs)


# ============================================================================
# Verification
# ============================================================================

if __name__ == '__main__':
    print("=" * 78)
    print("  NC-SSM Vision Tasks: Lane Detection + Critical Object Spotting")
    print("  Audio KWS -> Visual COS | ICCV 2027")
    print("=" * 78)

    device = 'cpu'
    B = 2
    img = torch.rand(B, 3, 288, 288, device=device)

    print("\n--- Task Head Tests ---\n")

    # Lane Head
    tokens = torch.rand(B, 324, 24, device=device)  # 18*18=324 patches
    lh = LaneHead(d_model=24, n_lanes=4, n_anchors=18, n_grids=100)
    lane_out = lh(tokens)
    lh_params = sum(p.numel() for p in lh.parameters())
    print(f"  LaneHead:           {lh_params:>8,} params | "
          f"output={list(lane_out.shape)}")

    # Critical Object Head
    det_h = CriticalObjectHead(d_model=24, n_classes=5,
                                n_patches_h=18, n_patches_w=18)
    hm, off, sz = det_h(tokens)
    dh_params = sum(p.numel() for p in det_h.parameters())
    print(f"  CriticalObjectHead: {dh_params:>8,} params | "
          f"hm={list(hm.shape)} off={list(off.shape)} sz={list(sz.shape)}")

    print("\n--- Full Model Tests ---\n")

    models = [
        ('LaneDet-Nano',       create_lane_detector_nano),
        ('LaneDet-Tiny',       create_lane_detector_tiny),
        ('CriticalDet-Nano',   create_critical_detector_nano),
        ('CriticalDet-Tiny',   create_critical_detector_tiny),
        ('MultiTask-Tiny',     create_multitask_detector_tiny),
    ]

    header = (f"  {'Model':<22} | {'Total':>8} | {'Backbone':>8} | "
              f"{'Head':>8} | {'INT8':>7} | Task")
    print(header)
    print("  " + "-" * 82)

    for name, create_fn in models:
        model = create_fn(img_size=288).to(device)
        model.eval()

        total = sum(p.numel() for p in model.parameters())
        backbone_p = sum(p.numel() for p in model.backbone.parameters())
        head_p = total - backbone_p
        kb_int8 = total / 1024

        with torch.no_grad():
            out = model(img)

        if isinstance(out, dict):
            task = f"lane={list(out['lanes'].shape)} hm={list(out['heatmap'].shape)}"
        elif isinstance(out, tuple):
            task = f"hm={list(out[0].shape)}"
        else:
            task = f"lane={list(out.shape)}"

        print(f"  {name:<22} | {total:>8,} | {backbone_p:>8,} | "
              f"{head_p:>8,} | {kb_int8:>5.1f}KB | {task}")

    print("\n--- Audio KWS <-> Vision COS Analogy ---\n")
    analogy = [
        ("Task",        "Keyword Spotting (12 cls)",     "Critical Object Spotting (5 cls)"),
        ("Input",       "1s audio @ 16kHz",              "288x288 image"),
        ("Backbone",    "NanoMamba NC-SSM (7.4K)",       "NanoMamba-V NC-SSM (~13K)"),
        ("Head",        "Linear(d, 12) = 252p",          "CenterNet head = ~1K p"),
        ("Noise",       "Factory/Babble -15dB",          "Tunnel/Night/Fog"),
        ("0-param",     "SS+Bypass + DualPCEN routing",  "Retinex+Bypass + DualNorm routing"),
        ("Key insight", "O(sigma^6)->O(sigma^2)",        "Same (vision domain)"),
    ]
    print(f"  {'':>12} | {'Audio (Interspeech 2026)':<35} | {'Vision (ICCV 2027)':<35}")
    print(f"  {'':>12} | {'-'*35} | {'-'*35}")
    for cat, audio, vision in analogy:
        print(f"  {cat:>12} | {audio:<35} | {vision:<35}")

    print(f"\n--- Benchmark Plan ---\n")
    benchmarks = [
        ("CULane",       "Lane Det",    "Normal/Night/Tunnel/Shadow/Dazzle"),
        ("ExDark",       "Object Det",  "10 classes, extreme low-light"),
        ("ACDC",         "Both",        "Fog/Night/Rain/Snow (Cityscapes fmt)"),
        ("BDD100K",      "Both",        "Day/Night/Tunnel subsets"),
    ]
    print(f"  {'Dataset':<12} | {'Task':<12} | Conditions")
    print(f"  {'-'*60}")
    for ds, task, cond in benchmarks:
        print(f"  {ds:<12} | {task:<12} | {cond}")

    print(f"\n--- Comparison Targets ---\n")
    baselines = [
        ("UFLD-v2",        "2.7M",   "Lane",  "Row-anchor SOTA"),
        ("CLRNet",         "9.0M",   "Lane",  "Cross-layer refine"),
        ("YOLOv10-nano",   "2.3M",   "Det",   "CNN lightweight SOTA"),
        ("IA-YOLO",        "~3M",    "Det",   "Adverse condition SOTA"),
        ("PE-YOLO",        "~4M",    "Det",   "Enhancement+Det (2023)"),
        ("VMamba-T+head",  "26M",    "Both",  "SSM baseline (no NC-SSM)"),
        ("Ours (Tiny)",    "~25K",   "Both",  "100x smaller, tunnel-robust"),
    ]
    print(f"  {'Model':<18} | {'Params':>8} | {'Task':<6} | Note")
    print(f"  {'-'*65}")
    for m, p, t, note in baselines:
        print(f"  {m:<18} | {p:>8} | {t:<6} | {note}")

    print(f"\n{'=' * 78}")
    print(f"  DONE. All tasks verified.")
    print(f"{'=' * 78}")
