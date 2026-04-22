# NC-Conv-SSM Vision (`ncconv/`)

**Noise-Conditioned Convolution + temporal NC-SSM for degradation-robust vision.**

This directory contains the Vision code that accompanies the NC-Conv-SSM
paper family. The core idea is to blend a **static** (fixed-weight, robust)
convolutional path and a **dynamic** (input-dependent, expressive) path
using a learned quality gate `sigma in [0,1]`:

```
h = sigma * h_dynamic + (1 - sigma) * h_static
```

Clean inputs route toward `sigma ~ 1` (dynamic), degraded inputs toward
`sigma ~ 0` (static). The gate is learned **without** explicit quality
labels, via corruption-augmented training.

## Target venues

- **ICCV 2027** -- primary conference target (`paper/iccv2027_ncssm_vision.tex`)
- **IEEE TIP 2027** -- journal extension (`paper/tip2027_ncconv.tex`)
- **BMVC 2026** (`paper/bmvc2026_ncconv.tex`)
- **ACCV 2026** -- Osaka (`paper/accv2026_ncconv.tex`)
- **CVPR 2027** (`paper/cvpr2027_ncconv.tex`)

## Headline results (Paper Table 2)

| Condition     | Std CNN (aug) | NC-Conv (aug) | Delta  |
|---            |---:           |---:           |---:    |
| Clean         | 89.0%         | 88.7%         | -0.3%  |
| Gaussian      | 86.3%         | 87.3%         | +1.0%  |
| Brightness    | 86.7%         | 87.3%         | +0.6%  |
| Contrast      | 72.8%         | 85.2%         | +12.4% |
| Fog           | 70.0%         | 83.5%         | +13.5% |
| Impulse       | 76.2%         | 79.1%         | +2.9%  |
| **Avg corrupt** | **78.4%**   | **84.5%**     | **+6.1%** |

Full scale / tunnel-video / CULane tables live in `results.py`.

## Files

| File | What |
|---|---|
| `models.py`       | Architectures: StandardCNN, NC-Conv v7 (per-sample sigma), NC-Conv v8 (per-spatial sigma), BiTemporalNCSBlock, VideoModelBiNC, LaneDetector. |
| `corruption.py`   | 5 corruption families mapped to audio-noise analogs (gaussian / brightness / contrast / fog / impulse). |
| `data.py`         | CIFAR-10 loaders, TunnelVideoDataset (8-frame tunnel simulation), CULaneFiles. |
| `results.py`      | Hardcoded paper results (CIFAR-10-C, temporal Bi-NC-SSM, scale ablation, CULane, per-spatial sigma). |
| `demo.py`         | Interactive Streamlit demo with live sigma-map visualisation. |
| `reproduce_vision.py` | One-command reviewer reproduction script (see below). |

## Quick start

```python
from ncconv.models import (
    StandardCNN, make_ncconv_net, NCConvBlock, NCConvBlockSpatial,
)
from ncconv.data import get_cifar10_loaders, TunnelVideoDataset
from ncconv.corruption import apply_corruption, random_corruption_batch
from ncconv.results import print_all_results

std_cnn = StandardCNN(n_classes=10)           # 251K params
nc_v7   = make_ncconv_net(NCConvBlock)        # 253K params (per-sample sigma)
nc_v8   = make_ncconv_net(NCConvBlockSpatial) # 254K params (per-spatial sigma)
print_all_results()
```

## Reproducing the paper

A reviewer-focused one-command script lives at `ncconv/reproduce_vision.py`:

```bash
# From the repository root:
python -m ncconv.reproduce_vision                # smoke test (~15 s)
python -m ncconv.reproduce_vision --full         # + tunnel & CULane tables
python -m ncconv.reproduce_vision --eval         # + CIFAR-10-C eval
                                                 #   (needs a trained ckpt at
                                                 #   checkpoints_vision/ncconv_cifar10_best.pt)
python -m ncconv.reproduce_vision --help
```

The script:

1. Checks the environment (`torch`, `torchvision`, `numpy`).
2. Instantiates StandardCNN, NC-Conv v7, NC-Conv v8 and verifies parameter
   counts match the paper (251K / 253K / 254K, within +/-1%).
3. Runs a forward-pass smoke test and checks that `sigma in [0, 1]` under
   every corruption family.
4. Prints the paper Table 2 (CIFAR-10 / CIFAR-10-C).
5. With `--full`: prints the tunnel-video Bi-NC-SSM and CULane lane-detection
   paper tables (from `results.py`).
6. With `--eval` + a trained checkpoint: measures CIFAR-10-C locally and
   compares to the paper, exiting non-zero if mean deviation > `--tol` (%pts).

**Full training** (to produce the `--eval` checkpoint) is done via the
Colab notebook `NC_SSM_Vision_Colab.ipynb` at the repo root (CIFAR-10 +
corruption-augmented; canonical hyperparameters in `results.py`).

### Interactive demo

```bash
pip install streamlit opencv-python
streamlit run ncconv/demo.py -- --mode streamlit
# or: python -m ncconv.demo --mode sigma   # terminal sigma analysis
```

## MCU deployment

The 253K NC-Conv network is INT8-friendly: 253 KB static weights fit on an
STM32H743 (Cortex-M7, 480 MHz, 1 MB SRAM, ~$8) at roughly 28 FPS, 35 ms
latency, 0.5 W -- see `paper/iccv2027_ncssm_vision.tex` Section 6 and the
ACCV companion paper for the deployment analysis.

## Citation

Please cite at least one of the NC-Conv-SSM Vision papers when using this
code:

```bibtex
@inproceedings{choi2027ncconv_iccv,
  author    = {Jin Ho Choi},
  title     = {Noise-Conditioned Convolution: Bridging Dynamic Expressiveness
               and Static Robustness for Degraded Vision},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision
               (ICCV)},
  year      = {2027},
  note      = {submitted}
}

@article{choi2027ncconv_tip,
  author  = {Jin Ho Choi},
  title   = {NC-Conv-SSM: A Unified Noise-Conditioned Framework for
             Degradation-Robust Visual Recognition},
  journal = {IEEE Transactions on Image Processing},
  year    = {2027},
  note    = {submitted}
}
```

Companion venues (overlapping material): BMVC 2026, ACCV 2026, CVPR 2027.
Full BibTeX entries in `COMMERCIAL_LICENSE.md` Section 6.

## License

This directory ships under a **dual license**:

- **Academic / non-commercial**: free for research, teaching, thesis,
  benchmark reproduction -- see [`LICENSE`](LICENSE).
- **Commercial** (product integration, ADAS / drone / surveillance
  deployment, MCU / NPU / FPGA / ASIC implementations sold to third
  parties): a separate written license is required -- see
  [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) and contact
  `jinhochoi@smartear.co.kr`.

Technologies embodied in this code are subject to pending patents in the
Republic of Korea and the United States (NanoMamba / NC-SSM family,
including the NC-Conv spatial variant and the bidirectional temporal
NC-SSM for video). The academic license grants no commercial patent rights.

The root-level `LICENSE` governs the rest of the repository (audio NC-SSM,
NC-TCN); where both apply to `ncconv/` code, this directory's `LICENSE`
controls.
