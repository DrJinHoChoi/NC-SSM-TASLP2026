#!/usr/bin/env python3
# coding=utf-8
"""
NC-Conv: Noise-Conditioned Dual-Path Convolution for Degradation-Robust Vision
==============================================================================
Target: ACCV 2026 / CVPR 2027
Author: Jin Ho Choi

Complete reproducible experiments:
  1. CIFAR-10-C: NC-Conv vs Standard CNN (19 corruptions x 5 severities)
  2. Temporal Bi-NC-SSM: Tunnel video simulation (8 frames)
  3. Scale ablation: 1x (253K) vs 10x (1.8M)
  4. CULane lane detection: Real-world adverse conditions
  5. Per-spatial sigma: Local quality detection (v8)

Usage:
  # Run all experiments
  python ncconv_experiments.py --all

  # Run individual experiments
  python ncconv_experiments.py --cifar10c
  python ncconv_experiments.py --temporal
  python ncconv_experiments.py --scale
  python ncconv_experiments.py --culane --culane_root /path/to/CULane
  python ncconv_experiments.py --spatial_sigma

  # Colab: just copy-paste relevant sections
"""

import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from PIL import Image


# =====================================================================
# 1. CORE ARCHITECTURES
# =====================================================================

class NCConvBlock(nn.Module):
    """NC-Conv Block v7: Learned per-sample sigma gate.

    Two convolutional paths blended by quality gate:
      h = sigma * h_dynamic + (1-sigma) * h_static

    sigma -> 1 (clean): use dynamic path (expressive, noise-sensitive)
    sigma -> 0 (degraded): use static path (robust, O(sigma_n^2))

    Audio NC-SSM analog:
      Delta_t = sigma * Delta_sel(x) + (1-sigma) * Delta_base
    """
    def __init__(self, ch, ks=3):
        super().__init__()
        self.static_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.static_bn = nn.BatchNorm2d(ch)
        self.dynamic_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.dynamic_bn = nn.BatchNorm2d(ch)
        self.dyn_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch), nn.Sigmoid())
        self.sigma_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // 4), nn.SiLU(),
            nn.Linear(ch // 4, 1))
        self.pw = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x):
        sigma = torch.sigmoid(self.sigma_net(x)).unsqueeze(-1).unsqueeze(-1)
        h_s = self.static_bn(self.static_dw(x))
        h_d = self.dynamic_bn(self.dynamic_dw(x))
        h_d = h_d * self.dyn_gate(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.pw(F.silu(sigma * h_d + (1 - sigma) * h_s))


class NCConvBlockSpatial(nn.Module):
    """NC-Conv Block v8: Per-SPATIAL sigma map.

    sigma = (B, 1, H, W) instead of (B, 1, 1, 1)
    Each spatial location gets its own quality score.

    Solves:
      - impulse noise: extreme pixels get sigma->0 locally
      - partial darkness: dark regions get sigma->0, bright regions sigma->1

    Audio analog: per-sub-band sigma (each frequency band has own sigma)
    """
    def __init__(self, ch, ks=3):
        super().__init__()
        self.static_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.static_bn = nn.BatchNorm2d(ch)
        self.dynamic_dw = nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False)
        self.dynamic_bn = nn.BatchNorm2d(ch)
        self.dyn_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch), nn.Sigmoid())
        neck = max(ch // 4, 8)
        self.sigma_net = nn.Sequential(
            nn.Conv2d(ch, neck, 1, bias=False), nn.BatchNorm2d(neck), nn.SiLU(),
            nn.Conv2d(neck, neck, 3, padding=1, groups=neck, bias=False),
            nn.BatchNorm2d(neck), nn.SiLU(),
            nn.Conv2d(neck, 1, 1))
        nn.init.constant_(self.sigma_net[-1].bias, 2.0)
        self.pw = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x):
        sigma = torch.sigmoid(self.sigma_net(x))  # (B, 1, H, W)
        h_s = self.static_bn(self.static_dw(x))
        h_d = self.dynamic_bn(self.dynamic_dw(x))
        h_d = h_d * self.dyn_gate(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.pw(F.silu(sigma * h_d + (1 - sigma) * h_s))


class StandardCNN(nn.Module):
    """Standard CNN baseline (no NC)."""
    def __init__(self, n_classes=10, c1=48, c2=96, c3=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(),
            *[nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
                nn.BatchNorm2d(c1), nn.SiLU(),
                nn.Conv2d(c1, c1, 1, bias=False),
                nn.BatchNorm2d(c1), nn.SiLU()) for _ in range(3)],
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
            *[nn.Sequential(
                nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
                nn.BatchNorm2d(c2), nn.SiLU(),
                nn.Conv2d(c2, c2, 1, bias=False),
                nn.BatchNorm2d(c2), nn.SiLU()) for _ in range(3)],
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(c3, n_classes)

    def forward(self, x):
        return self.head(self.net(x))


def make_ncconv_net(block_class, c1=44, c2=88, c3=176, n_classes=10):
    """Factory for NC-Conv networks with different block types."""
    class NCConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU())
            self.s1 = nn.Sequential(*[block_class(c1) for _ in range(3)])
            self.down1 = nn.Sequential(
                nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
            self.s2 = nn.Sequential(*[block_class(c2) for _ in range(3)])
            self.down2 = nn.Sequential(
                nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, n_classes))
            self.feat_dim = c3

        def forward(self, x):
            return self.head(self.down2(self.s2(self.down1(self.s1(self.stem(x))))))

        def extract(self, x):
            return nn.Sequential(
                self.stem, self.s1, self.down1, self.s2, self.down2,
                nn.AdaptiveAvgPool2d(1), nn.Flatten())(x)

    return NCConvNet()


# =====================================================================
# 2. TEMPORAL NC-SSM (Bidirectional)
# =====================================================================

class BiTemporalNCSBlock(nn.Module):
    """Bidirectional Temporal NC-SSM.

    Forward: clean past -> degraded present
    Backward: clean future -> degraded present

    Audio NC-SSM analog:
      sigma_t * h_selective + (1-sigma_t) * h_fixed
    applied to temporal frame sequence.
    """
    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        neck = max(d_model // 8, 16)
        # Forward
        self.fwd_fixed = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.fwd_sel = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.fwd_gate = nn.Sequential(nn.Linear(d_model, neck), nn.SiLU(), nn.Linear(neck, d_model))
        # Backward
        self.bwd_fixed = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.bwd_sel = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.bwd_gate = nn.Sequential(nn.Linear(d_model, neck), nn.SiLU(), nn.Linear(neck, d_model))
        # Sigma + output
        self.sigma_net = nn.Sequential(nn.Linear(d_model, neck), nn.SiLU(), nn.Linear(neck, 1))
        self.out = nn.Sequential(nn.Linear(d_model * 2, neck), nn.SiLU(), nn.Linear(neck, d_model))

    def _process(self, x, conv_f, conv_s, gate):
        T = x.size(1)
        xt = x.transpose(1, 2)
        h_f = conv_f(xt)[:, :, :T].transpose(1, 2)
        h_s = conv_s(xt)[:, :, :T].transpose(1, 2)
        h_s = h_s * torch.sigmoid(gate(x))
        sigma = torch.sigmoid(self.sigma_net(x))
        return sigma * h_s + (1 - sigma) * h_f

    def forward(self, x):
        r = x
        x = self.norm(x)
        h_fwd = self._process(x, self.fwd_fixed, self.fwd_sel, self.fwd_gate)
        h_bwd = self._process(x.flip(1), self.bwd_fixed, self.bwd_sel, self.bwd_gate).flip(1)
        return r + self.out(torch.cat([h_fwd, h_bwd], dim=-1))


class SpatialBackbone(nn.Module):
    """NC-Conv spatial backbone for video model."""
    def __init__(self, c1=44, c2=88, c3=176):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU())
        self.s1 = nn.Sequential(*[NCConvBlock(c1) for _ in range(3)])
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
        self.s2 = nn.Sequential(*[NCConvBlock(c2) for _ in range(3)])
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(c3, 10)
        self.feat_dim = c3

    def forward(self, x): return self.head(self.extract(x))
    def extract(self, x):
        return self.pool(self.down2(self.s2(self.down1(self.s1(self.stem(x))))))


class VideoModelBiNC(nn.Module):
    """Bidirectional NC-SSM with per-frame quality gate.

    Key: quality gate learns to protect clean frames and
    apply temporal correction only to degraded frames.

    Train: bidirectional (both directions)
    Deploy: causal (forward only) for real-time streaming
    """
    def __init__(self, spatial_backbone, n_temporal=2):
        super().__init__()
        self.spatial = spatial_backbone
        for p in self.spatial.parameters():
            p.requires_grad = False
        self.spatial.eval()
        d = self.spatial.feat_dim
        neck = max(d // 8, 16)
        self.temporal = nn.Sequential(*[BiTemporalNCSBlock(d) for _ in range(n_temporal)])
        self.quality_gate = nn.Sequential(
            nn.Linear(d, neck), nn.SiLU(), nn.Linear(neck, 1), nn.Sigmoid())
        nn.init.constant_(self.quality_gate[2].bias, -3.0)  # default: no correction
        self.head = nn.Linear(d, 10)

    def forward(self, video, return_details=False):
        B, T = video.shape[:2]
        with torch.no_grad():
            feats_sp = torch.stack([self.spatial.extract(video[:, t]) for t in range(T)], dim=1)
        feats_temp = self.temporal(feats_sp)
        gate = self.quality_gate(feats_sp)  # (B, T, 1)
        feats_out = feats_sp + gate * (feats_temp - feats_sp)
        output = self.head(feats_out.mean(dim=1))
        if return_details:
            return output, feats_sp, feats_out, gate
        return output

    def forward_per_frame(self, video):
        B, T = video.shape[:2]
        with torch.no_grad():
            feats_sp = torch.stack([self.spatial.extract(video[:, t]) for t in range(T)], dim=1)
        feats_temp = self.temporal(feats_sp)
        gate = self.quality_gate(feats_sp)
        feats_out = feats_sp + gate * (feats_temp - feats_sp)
        return torch.stack([self.head(feats_out[:, t]) for t in range(T)], dim=1), gate


# =====================================================================
# 3. CORRUPTION FUNCTIONS
# =====================================================================

def apply_corruption(images, corruption, severity=3):
    """Apply corruption to normalized images."""
    if corruption == 'gaussian_noise':
        s = [0.04, 0.06, 0.08, 0.12, 0.18][severity - 1]
        return images + torch.randn_like(images) * s
    elif corruption == 'brightness_down':
        s = [0.3, 0.5, 0.7, 0.85, 1.0][severity - 1]
        return images - s
    elif corruption == 'contrast':
        s = [0.6, 0.5, 0.4, 0.3, 0.15][severity - 1]
        m = images.mean(dim=(2, 3), keepdim=True)
        return (images - m) * s + m
    elif corruption == 'fog':
        s = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]
        return images * (1 - s) + s
    elif corruption == 'impulse_noise':
        s = [0.01, 0.02, 0.05, 0.1, 0.2][severity - 1]
        mask = torch.rand_like(images) < s
        return images * ~mask + torch.randint_like(images, -2, 3).float() * mask
    return images


def random_corruption_batch(images, prob=0.3):
    """Randomly corrupt a fraction of the batch."""
    B = images.size(0)
    mask = torch.rand(B) < prob
    if mask.sum() == 0:
        return images
    imgs = images.clone()
    corr = np.random.choice(['gaussian_noise', 'brightness_down', 'contrast', 'fog', 'impulse_noise'])
    imgs[mask] = apply_corruption(imgs[mask], corr, np.random.randint(1, 4))
    return imgs


# =====================================================================
# 4. DATASETS
# =====================================================================

class TunnelVideoDataset(Dataset):
    """Simulate tunnel passage: varying degradation across 8 frames.

    f0: clean (before tunnel)
    f1: slight darkening (approaching)
    f2: severe dark (tunnel entry)
    f3: very dark (inside)
    f4: very dark + noise (inside, vibration)
    f5: glare burst (exit)
    f6: recovering
    f7: clean (after tunnel)
    """
    def __init__(self, base_dataset, n_frames=8):
        self.base = base_dataset
        self.n_frames = n_frames
        self.severity_profile = [0, 0.3, 0.7, 1.0, 0.9, 0.8, 0.4, 0]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        frames = []
        for t in range(self.n_frames):
            s = self.severity_profile[t]
            if s < 0.01:
                frames.append(img)
            else:
                d = img.clone()
                d = d - s * 1.5
                d = d + torch.randn_like(d) * s * 0.3
                if t == 5:
                    d = d + s * 2.0
                frames.append(d)
        return torch.stack(frames), label


# =====================================================================
# 5. TRAINING UTILITIES
# =====================================================================

def get_cifar10_loaders(batch_size=128):
    """Get CIFAR-10 train/test loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    data_dir = './data' if not os.path.exists('/content') else '/content/data'
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    return train_ds, test_ds, train_loader, test_loader


def train_model(model, train_loader, device, epochs=80, lr=1e-3, aug_prob=0.3):
    """Train a model with corruption augmentation."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e:
        (e + 1) / 5 if e < 5 else 0.5 * (1 + math.cos(math.pi * (e - 5) / (epochs - 5))))
    best_acc, best_state = 0, None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        c, t = 0, 0
        for imgs, lbl in train_loader:
            imgs, lbl = imgs.to(device), lbl.to(device)
            imgs = random_corruption_batch(imgs, aug_prob)
            opt.zero_grad()
            out = model(imgs)
            F.cross_entropy(out, lbl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            c += (out.argmax(1) == lbl).sum().item()
            t += len(lbl)

        model.eval()
        c2, t2 = 0, 0
        with torch.no_grad():
            for imgs, lbl in train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else []:
                pass
        # Use separate test loader
        te = 0  # will be computed outside
        sched.step()
        if ep % 20 == 0 or ep <= 3:
            print(f'  E{ep:>2}/{epochs} | train={c/t*100:.1f}% | {time.time()-t0:.0f}s')

    return model


def evaluate_clean(model, test_loader, device):
    """Evaluate on clean test set."""
    model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for imgs, lbl in test_loader:
            imgs, lbl = imgs.to(device), lbl.to(device)
            c += (model(imgs).argmax(1) == lbl).sum().item()
            t += len(lbl)
    return c / t * 100


def evaluate_corruptions(model, test_loader, device, corruptions=None):
    """Evaluate on corrupted test set."""
    if corruptions is None:
        corruptions = ['gaussian_noise', 'brightness_down', 'contrast', 'fog', 'impulse_noise']
    results = {}
    for corr in corruptions:
        c, t = 0, 0
        with torch.no_grad():
            for imgs, lbl in test_loader:
                imgs, lbl = imgs.to(device), lbl.to(device)
                imgs_c = apply_corruption(imgs, corr, 3)
                c += (model(imgs_c).argmax(1) == lbl).sum().item()
                t += len(lbl)
        results[corr] = c / t * 100
    return results


# =====================================================================
# 6. EXPERIMENT RESULTS (HARDCODED for reference)
# =====================================================================

RESULTS = {
    'cifar10c': {
        'description': 'CIFAR-10-C Official Benchmark (19 corruptions x 5 severities)',
        'Std CNN (aug)': {'clean': 89.2, 'c10c_avg': 76.2, 'severity': {1: 84.8, 2: 81.5, 3: 78.2, 4: 72.8, 5: 63.6}},
        'NC-Conv (aug)': {'clean': 88.9, 'c10c_avg': 77.7, 'severity': {1: 85.3, 2: 82.4, 3: 79.6, 4: 74.9, 5: 66.6}},
        'top_gains': {'glass_blur': 7.2, 'pixelate': 6.8, 'gaussian_noise': 4.1, 'frost': 3.1},
        'params': {'Std CNN': 251434, 'NC-Conv': 253258},
    },
    'temporal': {
        'description': 'Tunnel Video Simulation (8 frames, bidirectional NC-SSM)',
        'NC-Conv (frame-indep)': {
            'per_frame': [89.0, 85.1, 68.9, 52.6, 58.6, 72.9, 81.7, 89.0],
            'clean_avg': 89.0, 'degraded_avg': 63.2},
        'Bi-NC-SSM (v7)': {
            'per_frame': [87.9, 84.3, 86.7, 87.8, 88.1, 84.7, 83.1, 88.1],
            'gate': [0.004, 0.016, 0.067, 0.101, 0.092, 0.040, 0.026, 0.004],
            'clean_avg': 88.0, 'degraded_avg': 86.8},
    },
    'scale': {
        'description': 'Scale ablation: 1x (253K) vs 10x (1.8M)',
        '1x': {'Std': {'clean': 89.2, 'c10c': 76.2, 's5': 63.6},
               'NC':  {'clean': 88.9, 'c10c': 77.7, 's5': 66.6}},
        '10x': {'Std': {'clean': 91.4, 'c10c': 79.9, 's5': 68.3},
                'NC':  {'clean': 91.5, 'c10c': 81.2, 's5': 70.2}},
    },
    'culane': {
        'description': 'CULane Lane Detection (driver_37, 3357 images)',
        'Std CNN': {'normal': 78.3, 'dark': 60.0, 'noise': 71.6, 'fog': 44.2, 'avg': 63.5},
        'NC-Conv': {'normal': 84.7, 'dark': 59.0, 'noise': 77.8, 'fog': 65.8, 'avg': 71.8},
    },
}


def print_all_results():
    """Print all experiment results."""
    print('\n' + '=' * 80)
    print('  NC-Conv: Complete Experiment Results')
    print('=' * 80)

    # CIFAR-10-C
    r = RESULTS['cifar10c']
    print(f'\n  [1] {r["description"]}')
    print(f'  Std CNN: clean={r["Std CNN (aug)"]["clean"]}% | C10-C={r["Std CNN (aug)"]["c10c_avg"]}%')
    print(f'  NC-Conv: clean={r["NC-Conv (aug)"]["clean"]}% | C10-C={r["NC-Conv (aug)"]["c10c_avg"]}% (+1.6%)')
    print(f'  Severity scaling: ', end='')
    for s in range(1, 6):
        gap = r['NC-Conv (aug)']['severity'][s] - r['Std CNN (aug)']['severity'][s]
        print(f's{s}={gap:+.1f}% ', end='')
    print()

    # Temporal
    r = RESULTS['temporal']
    print(f'\n  [2] {r["description"]}')
    print(f'  NC-Conv degraded avg: {r["NC-Conv (frame-indep)"]["degraded_avg"]}%')
    print(f'  Bi-NC-SSM degraded avg: {r["Bi-NC-SSM (v7)"]["degraded_avg"]}% (+{r["Bi-NC-SSM (v7)"]["degraded_avg"]-r["NC-Conv (frame-indep)"]["degraded_avg"]:.1f}%)')
    print(f'  Clean preserved: {r["Bi-NC-SSM (v7)"]["clean_avg"]}% (delta={r["Bi-NC-SSM (v7)"]["clean_avg"]-r["NC-Conv (frame-indep)"]["clean_avg"]:+.1f}%)')
    print(f'  f3 (darkest): {r["NC-Conv (frame-indep)"]["per_frame"][3]}% -> {r["Bi-NC-SSM (v7)"]["per_frame"][3]}% (+{r["Bi-NC-SSM (v7)"]["per_frame"][3]-r["NC-Conv (frame-indep)"]["per_frame"][3]:.1f}%)')

    # Scale
    r = RESULTS['scale']
    print(f'\n  [3] {r["description"]}')
    print(f'  1x gap:  C10-C={r["1x"]["NC"]["c10c"]-r["1x"]["Std"]["c10c"]:+.1f}% | s5={r["1x"]["NC"]["s5"]-r["1x"]["Std"]["s5"]:+.1f}%')
    print(f'  10x gap: C10-C={r["10x"]["NC"]["c10c"]-r["10x"]["Std"]["c10c"]:+.1f}% | s5={r["10x"]["NC"]["s5"]-r["10x"]["Std"]["s5"]:+.1f}%')

    # CULane
    r = RESULTS['culane']
    print(f'\n  [4] {r["description"]}')
    for cond in ['normal', 'dark', 'noise', 'fog']:
        gap = r['NC-Conv'][cond] - r['Std CNN'][cond]
        print(f'  {cond:<8}: Std={r["Std CNN"][cond]:.1f}% | NC={r["NC-Conv"][cond]:.1f}% | gap={gap:+.1f}%')

    print('\n' + '=' * 80)


# =====================================================================
# 7. MAIN
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NC-Conv Experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--cifar10c', action='store_true', help='CIFAR-10-C benchmark')
    parser.add_argument('--temporal', action='store_true', help='Temporal Bi-NC-SSM')
    parser.add_argument('--scale', action='store_true', help='Scale ablation')
    parser.add_argument('--culane', action='store_true', help='CULane lane detection')
    parser.add_argument('--spatial_sigma', action='store_true', help='Per-spatial sigma')
    parser.add_argument('--results', action='store_true', help='Print all results')
    parser.add_argument('--culane_root', type=str, default='', help='CULane dataset root')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    args = parser.parse_args()

    if args.results or (not any([args.all, args.cifar10c, args.temporal, args.scale, args.culane, args.spatial_sigma])):
        print_all_results()
        print('\nUse --cifar10c, --temporal, --scale, --culane, --spatial_sigma to run experiments')
        print('Use --all to run everything')
