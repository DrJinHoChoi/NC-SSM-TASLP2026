#!/usr/bin/env python3
"""
NanoMamba Full ARM Deployment Analysis
======================================
All models: NanoMamba-Tiny-DualPCEN, NanoMamba-Matched-DualPCEN,
            BC-ResNet-1, DS-CNN-S

Computes: MACs, Memory, Latency, Power, Battery Life
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from nanomamba import NanoMamba
from train_colab import DSCNN_S, BCResNet


# ============================================================
# Precise MAC Counter using hooks
# ============================================================
class MACCounter:
    """Count MACs by hooking into every layer."""
    def __init__(self):
        self.total_macs = 0
        self.layer_macs = {}
        self.hooks = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            macs = 0
            if isinstance(module, nn.Linear):
                # MACs = in_features * out_features (* batch)
                in_f = module.in_features
                out_f = module.out_features
                if isinstance(input[0], torch.Tensor):
                    batch_elements = input[0].numel() // in_f
                else:
                    batch_elements = 1
                macs = in_f * out_f * batch_elements
                if module.bias is not None:
                    macs += out_f * batch_elements

            elif isinstance(module, nn.Conv2d):
                if isinstance(output, torch.Tensor):
                    out_h, out_w = output.shape[2], output.shape[3]
                else:
                    out_h, out_w = 1, 1
                k_h, k_w = module.kernel_size
                in_c = module.in_channels // module.groups
                out_c = module.out_channels
                macs = k_h * k_w * in_c * out_c * out_h * out_w
                if module.bias is not None:
                    macs += out_c * out_h * out_w

            elif isinstance(module, nn.Conv1d):
                if isinstance(output, torch.Tensor):
                    out_l = output.shape[2]
                else:
                    out_l = 1
                k = module.kernel_size[0]
                in_c = module.in_channels // module.groups
                out_c = module.out_channels
                macs = k * in_c * out_c * out_l
                if module.bias is not None:
                    macs += out_c * out_l

            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if isinstance(input[0], torch.Tensor):
                    macs = input[0].numel() * 2  # mul + add per element

            elif isinstance(module, (nn.LayerNorm, nn.InstanceNorm1d)):
                if isinstance(input[0], torch.Tensor):
                    macs = input[0].numel() * 4  # mean, var, sub, div

            if macs > 0:
                self.total_macs += macs
                if name not in self.layer_macs:
                    self.layer_macs[name] = 0
                self.layer_macs[name] += macs
        return hook

    def register(self, model, prefix=''):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d,
                                   nn.BatchNorm2d, nn.BatchNorm1d,
                                   nn.LayerNorm, nn.InstanceNorm1d)):
                full_name = f"{prefix}.{name}" if prefix else name
                h = module.register_forward_hook(self._hook_fn(full_name))
                self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.total_macs = 0
        self.layer_macs = {}


def count_macs(model, input_tensor):
    """Count total MACs for a forward pass."""
    counter = MACCounter()
    counter.register(model)
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    counter.remove()
    return counter.total_macs, counter.layer_macs


def count_peak_memory(model, input_tensor):
    """Estimate peak activation memory in bytes (FP32)."""
    activations = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.numel() * 4)  # FP32 = 4 bytes

    for module in model.modules():
        if not isinstance(module, type(model)):
            h = module.register_forward_hook(hook_fn)
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    peak = max(activations) if activations else 0
    total = sum(activations)
    return peak, total


# ============================================================
# Create models
# ============================================================
print("=" * 75)
print("  NanoMamba ARM Deployment Analysis - All Models")
print("=" * 75)
print()

models = {}

# NanoMamba-Tiny-DualPCEN
m = NanoMamba(n_mels=40, n_classes=12, d_model=16, d_state=4,
              d_conv=3, expand=1.5, n_layers=2, use_dual_pcen=True)
models['NanoMamba-Tiny-DualPCEN'] = (m, 'raw')

# NanoMamba-Matched-DualPCEN
m = NanoMamba(n_mels=40, n_classes=12, d_model=21, d_state=5,
              d_conv=3, expand=1.5, n_layers=2, use_dual_pcen=True)
models['NanoMamba-Matched-DualPCEN'] = (m, 'raw')

# DS-CNN-S
m = DSCNN_S(n_classes=12)
models['DS-CNN-S'] = (m, 'mel')

# BC-ResNet-1
m = BCResNet(n_classes=12, scale=1)
models['BC-ResNet-1'] = (m, 'mel')

# ============================================================
# Compute MACs, Memory for each model
# ============================================================
# Inputs
raw_audio = torch.randn(1, 16000)     # 1 sec @ 16kHz
mel_input = torch.randn(1, 40, 101)   # 40 mels x 101 frames

# STFT overhead (shared for all, but NanoMamba does it internally)
n_fft, hop, T, n_freq = 512, 160, 101, 257
stft_macs = T * (n_fft * 9 // 2) + T * n_freq  # FFT + magnitude

results = {}
for name, (model, input_type) in models.items():
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    inp = raw_audio if input_type == 'raw' else mel_input
    macs, layer_macs = count_macs(model, inp)
    peak_act, total_act = count_peak_memory(model, inp)

    # For CNN models, add STFT + mel extraction overhead
    if input_type == 'mel':
        mel_extract_macs = stft_macs + int(40 * 6.4 * T)  # STFT + sparse mel proj
        macs_with_fe = macs + mel_extract_macs
    else:
        macs_with_fe = macs  # NanoMamba includes STFT internally
        mel_extract_macs = 0

    results[name] = {
        'params': params,
        'macs_model': macs,
        'macs_fe': mel_extract_macs,
        'macs_total': macs_with_fe,
        'peak_act': peak_act,
        'total_act': total_act,
        'layer_macs': layer_macs,
    }

# ============================================================
# Print Results
# ============================================================

# --- 1. Parameter & MAC Summary ---
print("=" * 75)
print("  1. Parameter & MAC Comparison")
print("=" * 75)
print(f"  {'Model':<30} {'Params':>8} {'MACs':>12} {'M MACs':>8} {'MAC/P':>7}")
print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*8} {'-'*7}")
for name, r in results.items():
    m = r['macs_total']
    print(f"  {name:<30} {r['params']:>8,} {m:>12,} {m/1e6:>7.2f} {m/r['params']:>6.1f}")
print()

# --- 2. Memory ---
print("=" * 75)
print("  2. Memory Requirements (INT8 deployment)")
print("=" * 75)
print(f"  {'Model':<30} {'Wt(INT8)':>10} {'Wt(FP32)':>10} {'Act Peak':>10} {'Total RAM':>10}")
print(f"  {'':>30} {'(KB)':>10} {'(KB)':>10} {'(KB)':>10} {'(KB)':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for name, r in results.items():
    w8 = r['params'] / 1024
    w32 = r['params'] * 4 / 1024
    act = r['peak_act'] / 1024
    total = (r['params'] + r['peak_act']) / 1024
    print(f"  {name:<30} {w8:>9.1f} {w32:>9.1f} {act:>9.1f} {total:>9.1f}")
print()

# --- 3. ARM Latency & Power ---
arm_targets = [
    ("Cortex-M4 (STM32F4)",    168,  1, 120, "IoT/Wearable"),
    ("Cortex-M7 (STM32H7)",    480,  1, 200, "High-perf MCU"),
    ("Cortex-M33 (nRF5340)",   128,  1,  25, "BLE Audio"),
    ("Cortex-M55+Ethos-U55",   250,  8,  30, "AI MCU"),
    ("Cortex-M85",             320,  4,  80, "Next-gen MCU"),
]

print("=" * 75)
print("  3. ARM Latency (ms) per 1-second inference")
print("=" * 75)
header = f"  {'Processor':<25}"
for name in results:
    short = name.replace('NanoMamba-', 'NM-').replace('-DualPCEN', '')
    header += f" {short:>12}"
print(header)
print(f"  {'-'*25}" + f" {'-'*12}" * len(results))

for proc_name, mhz, mac_cyc, power_mw, desc in arm_targets:
    line = f"  {proc_name:<25}"
    for name, r in results.items():
        throughput = mhz * 1e6 * mac_cyc
        lat = r['macs_total'] / throughput * 1000
        line += f" {lat:>11.2f}"
    print(line)
print()

# --- 4. Energy per inference ---
print("=" * 75)
print("  4. Energy per Inference (uJ)")
print("=" * 75)
header = f"  {'Processor':<25}"
for name in results:
    short = name.replace('NanoMamba-', 'NM-').replace('-DualPCEN', '')
    header += f" {short:>12}"
print(header)
print(f"  {'-'*25}" + f" {'-'*12}" * len(results))

for proc_name, mhz, mac_cyc, power_mw, desc in arm_targets:
    line = f"  {proc_name:<25}"
    for name, r in results.items():
        throughput = mhz * 1e6 * mac_cyc
        lat_ms = r['macs_total'] / throughput * 1000
        energy = power_mw * lat_ms
        line += f" {energy:>11.1f}"
    print(line)
print()

# --- 5. Battery Life ---
print("=" * 75)
print("  5. Battery Life (days) — CR2032 (675 mWh), 1 inference/sec")
print("=" * 75)
header = f"  {'Processor':<25}"
for name in results:
    short = name.replace('NanoMamba-', 'NM-').replace('-DualPCEN', '')
    header += f" {short:>12}"
print(header)
print(f"  {'-'*25}" + f" {'-'*12}" * len(results))

for proc_name, mhz, mac_cyc, power_mw, desc in arm_targets:
    line = f"  {proc_name:<25}"
    for name, r in results.items():
        throughput = mhz * 1e6 * mac_cyc
        lat_ms = r['macs_total'] / throughput * 1000
        energy_uj = power_mw * lat_ms
        avg_mw = energy_uj / 1000 + 0.01  # + 0.01mW standby
        days = 675.0 / avg_mw / 24
        line += f" {days:>11.0f}"
    print(line)
print()

# --- 6. Detailed MAC breakdown per model ---
print("=" * 75)
print("  6. Top-10 MAC Layers per Model")
print("=" * 75)
for name, r in results.items():
    print(f"\n  [{name}] Total: {r['macs_total']:,} MACs ({r['macs_total']/1e6:.2f}M)")
    sorted_layers = sorted(r['layer_macs'].items(), key=lambda x: x[1], reverse=True)[:10]
    for lname, lmacs in sorted_layers:
        pct = lmacs / r['macs_total'] * 100
        print(f"    {lname:<50} {lmacs:>10,} ({pct:>5.1f}%)")

# --- 7. Efficiency Summary ---
print()
print("=" * 75)
print("  7. Efficiency Summary (Cortex-M7 @ 480 MHz)")
print("=" * 75)
print(f"  {'Model':<30} {'Params':>7} {'MACs':>8} {'Lat(ms)':>8} {'uJ':>8} {'Acc@0dB':>8} {'Acc/-15':>8}")
print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

# Average 0dB and -15dB accuracy (from noise-aug results)
acc_data = {
    'NanoMamba-Tiny-DualPCEN':    {'0dB': 85.4, '-15dB': 50.7, 'clean': 92.4},
    'NanoMamba-Matched-DualPCEN': {'0dB': 0, '-15dB': 0, 'clean': 0},  # pending
    'DS-CNN-S':                   {'0dB': 91.5, '-15dB': 64.2, 'clean': 96.4},
    'BC-ResNet-1':                {'0dB': 89.0, '-15dB': 63.1, 'clean': 95.3},
}

for name, r in results.items():
    m = r['macs_total']
    lat = m / 480e6 * 1000
    energy = 200 * lat
    acc = acc_data.get(name, {})
    a0 = f"{acc.get('0dB', 0):.1f}" if acc.get('0dB', 0) > 0 else "TBD"
    a15 = f"{acc.get('-15dB', 0):.1f}" if acc.get('-15dB', 0) > 0 else "TBD"
    print(f"  {name:<30} {r['params']:>7,} {m/1e6:>7.2f}M {lat:>7.2f} {energy:>7.1f} {a0:>8} {a15:>8}")

print()

# --- 8. SS+Bypass overhead ---
print("=" * 75)
print("  8. Spectral Subtraction + Bypass Overhead")
print("=" * 75)
ss_macs = n_freq * T * 3 + n_freq * 5  # magnitude ops on existing STFT
bypass_macs = n_freq * T                # SNR estimation for bypass gate
total_ss = ss_macs + bypass_macs
print(f"  SS processing:     {ss_macs:>10,} MACs")
print(f"  Bypass gate:       {bypass_macs:>10,} MACs")
print(f"  Total SS+Bypass:   {total_ss:>10,} MACs")
print()
print(f"  Overhead per model:")
for name, r in results.items():
    pct = total_ss / r['macs_total'] * 100
    print(f"    {name:<30} +{pct:.1f}%")
