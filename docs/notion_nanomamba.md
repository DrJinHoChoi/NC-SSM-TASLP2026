# NanoMamba KWS Engine

> **4.8KB AI that hears through noise.**
> Ultra-low-power keyword spotting engine for ARM Cortex-M.
> 10x fewer operations than CNN baselines. Battery life measured in years.

---

## Overview

NanoMamba is an ultra-compact, noise-robust keyword spotting (KWS) engine based on State Space Models (SSM). Designed for always-on voice interfaces in battery-powered devices.

| Spec | Value |
|------|-------|
| Model Size (INT8) | **4.8 KB** |
| Total Parameters | **4,957** |
| MACs per Inference | **0.45M** |
| Latency (Cortex-M7) | **0.94 ms** |
| Clean Accuracy | **92.4%** (GSC V2, 12-class) |
| Noise Robustness | Works at **-15 dB SNR** |
| Extra Noise Canceling HW | **Not required** (0-param SS+Bypass) |

---

## Why NanoMamba?

### The Problem
Traditional KWS models (DS-CNN, BC-ResNet) deliver high accuracy but demand heavy computation. In noisy environments (factory, street, wind), accuracy drops sharply and separate noise cancellation hardware is needed.

### Our Solution
NanoMamba uses a **Spectral-Aware State Space Model** that natively understands noise. Instead of processing every frequency band equally, it dynamically adjusts its temporal memory based on per-band SNR estimates:

- **High SNR frames** -> fast adaptation (propagate speech)
- **Low SNR frames** -> long memory (suppress noise)

This eliminates the need for a separate enhancement module.

---

## Key Technologies

### 1. Spectral-Aware SSM (SA-SSM)
SNR-modulated temporal dynamics. The SSM's selection parameters (dt, B) are directly conditioned on per-band noise estimates.

### 2. DualPCEN: Dual-Expert Noise Routing
Two complementary front-end experts, routed by spectral flatness (0 learnable params):
- **Expert 1** (high delta): Babble/speech-like noise specialist
- **Expert 2** (low delta): Factory/white/stationary noise specialist
- **Routing**: Automatic per-frame noise classification

### 3. Architectural Guarantees
Non-learnable safety constraints that **cannot be optimized away** during training:
- Delta-floor: SSM never freezes
- Epsilon-bypass: Information always flows
- B-gate floor: Input never fully blocked

### 4. Spectral Subtraction + SNR-Adaptive Bypass
Classical 0-parameter noise enhancement with intelligent bypass:
- Low SNR: Apply spectral subtraction (remove noise)
- High SNR: Bypass (preserve clean signal)
- **+23.8%p improvement** at -15dB white noise, **zero accuracy loss** on clean

---

## Performance

### Noise Robustness (Accuracy %)

**NanoMamba-Tiny-DualPCEN (4,957 params)**

| Noise Type | Clean | 0 dB | -15 dB |
|------------|-------|------|--------|
| Factory | 93.7 | 85.8 | 58.6 |
| White | 93.6 | 83.6 | 37.6 |
| Babble | 93.8 | 89.3 | 70.6 |
| Street | 93.8 | 83.3 | 58.2 |
| Pink | 93.8 | 84.8 | 28.3 |

### With SS+Bypass Enhancement (0 extra params)

| Noise Type | -15 dB Baseline | -15 dB + SS | Improvement |
|------------|----------------|-------------|-------------|
| White | 37.6% | **61.4%** | **+23.8%p** |
| Pink | 28.3% | **57.5%** | **+29.2%p** |
| Factory | 58.6% | **62.2%** | +3.6%p |
| Street | 58.2% | **59.9%** | +1.7%p |
| Babble | 70.6% | 70.5% | -0.1%p (harmless) |

Clean accuracy: **100% preserved** across all noise types.

---

## Competitive Comparison

### Accuracy vs Parameters

| Model | Params | Clean | Avg 0dB | Avg -15dB |
|-------|--------|-------|---------|-----------|
| **NanoMamba-Tiny** | **4,957** | 92.4% | 85.4% | 50.7% |
| BC-ResNet-1 | 7,464 | 95.3% | 89.0% | 63.1% |
| DS-CNN-S | 23,756 | 96.4% | 91.5% | 64.2% |

### Computational Efficiency

| Model | Params | MACs | Latency (M7) | Memory (INT8) |
|-------|--------|------|-------------|--------------|
| **NanoMamba-Tiny** | **4,957** | **0.45M** | **0.94 ms** | **23.8 KB** |
| BC-ResNet-1 | 7,464 | 4.62M | 9.63 ms | 102.0 KB |
| DS-CNN-S | 23,756 | 24.41M | 50.85 ms | 285.7 KB |

**NanoMamba vs BC-ResNet-1 (similar params):**
- **10.3x fewer MACs**
- **10.2x lower latency**
- **4.3x less RAM**

### SS+Bypass: NanoMamba Benefits More

| Model | White -15dB Gain | Factory 0dB Loss |
|-------|-----------------|-----------------|
| **NanoMamba** | **+23.8%p** | -2.1%p |
| DS-CNN-S | +1.3%p | -3.4%p |
| BC-ResNet-1 | +0.5%p | -7.9%p |

NanoMamba's DualPCEN routing integrates SS-enhanced signals far more effectively than CNN baselines.

---

## ARM Deployment

### Latency (ms per 1-sec inference)

| Processor | NanoMamba | BC-ResNet-1 | DS-CNN-S |
|-----------|----------|-------------|----------|
| Cortex-M4 (168 MHz) | **2.67** | 27.51 | 145.30 |
| Cortex-M7 (480 MHz) | **0.94** | 9.63 | 50.85 |
| Cortex-M33 (128 MHz) | **3.51** | 36.10 | 190.70 |
| Cortex-M55+Ethos (250 MHz) | **0.22** | 2.31 | 12.21 |

### Battery Life (CR2032 coin cell, 1 inference/sec)

| Processor | NanoMamba | BC-ResNet-1 | DS-CNN-S |
|-----------|----------|-------------|----------|
| Cortex-M33 (nRF5340) | **288 days** | 31 days | 6 days |
| Cortex-M7 (STM32H7) | **143 days** | 15 days | 3 days |
| Cortex-M55+Ethos | **1,680 days** | 355 days | 75 days |

### Memory Footprint

| Model | Weights (INT8) | Total RAM | Fits on |
|-------|---------------|-----------|---------|
| **NanoMamba** | **4.8 KB** | **23.8 KB** | Any Cortex-M |
| BC-ResNet-1 | 7.3 KB | 102.0 KB | Cortex-M4+ |
| DS-CNN-S | 23.2 KB | 285.7 KB | Cortex-M7+ |

---

## Target Applications

### Hearing Aids & Hearables
- Ultra-low power (288 days on CR2032)
- Noise robustness in real-world environments
- BLE audio compatible (nRF54 series)
- No separate noise cancellation chip needed

### Industrial IoT
- Factory noise robustness (-15dB operation)
- Voice commands in manufacturing environments
- Fits on smallest MCUs (4.8KB model)

### Smart Home
- Battery-powered voice sensors (doorbells, remotes)
- Always-on keyword detection
- Years of battery life

### Wearables
- Watch/band form factor compatible
- Sub-millisecond response
- Minimal memory footprint

---

## Technology Stack

```
Raw Audio (16kHz, 1sec)
    |
    v
[STFT] --> [SNR Estimator] --> per-band noise profile
    |              |
    v              v
[Mel Filterbank]  [DualPCEN Routing]
    |              |
    v              v
[Expert 1: Non-stationary] + [Expert 2: Stationary]
    |
    v
[Patch Projection] --> [SA-SSM Block x2] --> [Classifier]
                           |
                    SNR modulates dt, B
                    (noise-aware dynamics)
```

### Optional Front-End: SS+Bypass (0 params)
```
Audio --> [Spectral Subtraction] --+
  |                                |
  +---> [SNR Estimator] --> gate --+--> Enhanced Audio
         high SNR: bypass
         low SNR: apply SS
```

---

## Scalable Model Family

| Variant | Params | Target | Status |
|---------|--------|--------|--------|
| NanoMamba-Tiny-DualPCEN | 4,957 | Ultra-compact IoT | Ready |
| NanoMamba-Matched-DualPCEN | 7,402 | BC-ResNet-1 replacement | Training |
| NanoMamba-Small-DualPCEN | 12,355 | Higher accuracy | Available |

---

## Intellectual Property

- **Paper**: IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), submitted 2026
- **Patent**: Korean patent application filed (SA-SSM + DualPCEN + MoE routing)
- **License**: Free for academic/research use. Commercial license available.

---

## Contact

**SmartEar Co., Ltd.**
- Email: jinhochoi@smartear.co.kr
- Inventor: Jin Ho Choi, Ph.D.

---

*NanoMamba: When every microjoule counts and every word matters.*
