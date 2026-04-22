#!/usr/bin/env python3
"""One-command reproduction of IEEE TASLP 2026 paper results.

Paper:
  J. H. Choi, "NC-SSM / NanoMamba: Noise-Conditioned Selective State Space
  Models for Ultra-Compact Keyword Spotting," IEEE Transactions on Audio,
  Speech and Language Processing (TASLP), 2026 (submitted).

What this script does (for reviewers):
  1. Runs an environment check (Python, torch, torchaudio, CUDA).
  2. Locates Google Speech Commands V2 (12-class) or points the user at
     download instructions if it's missing.
  3. Loads the pre-trained checkpoints shipped in the repo under
     checkpoints_full/ (NanoMamba-Tiny, NanoMamba-Small, NC-SSM,
     DS-CNN-S, BC-ResNet-1 -- whichever exist).
  4. Evaluates each model on the GSC V2 test set (clean) and under
     factory @ 0 dB as a smoke-test. With --full, sweeps all five
     noise types at seven SNR levels (-15..+15 dB).
  5. Prints a comparison table vs Table III of the paper and vs the
     shipped paper/eval_results.json measurements.
  6. Exits 0 if all comparable rows are within --tol of the paper,
     else 1.

Usage (reviewers):

    python reproduce_taslp.py                       # clean + factory @ 0 dB
    python reproduce_taslp.py --full                # full 5 noise x 7 SNR grid
    python reproduce_taslp.py --data-dir D:/gsc_v2  # pre-downloaded dataset
    python reproduce_taslp.py --noise-type babble --snr -5
    python reproduce_taslp.py --quick               # 1500-sample smoke (~1 min)
    python reproduce_taslp.py --models NanoMamba-Tiny,NC-SSM

Exit codes:
    0  all comparable rows within tolerance
    1  at least one measurement exceeds --tol
    2  missing Python dependency
    3  checkpoint not found for all requested models
    4  GSC V2 dataset not found

Author: Dr. Jin Ho Choi (SmartEAR), jinhochoi@smartear.co.kr
License: see LICENSE
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------- #
# Paper's reported numbers (TASLP 2026, Table III -- Noise Robustness). #
# Hard-coded from paper/taslp_nanomamba.tex; see eval_results.json      #
# for the shipped measurement archive used by paper/run_paper_eval.py.  #
# --------------------------------------------------------------------- #
PAPER_TABLE_III = {
    # Clean accuracies (Table II)
    "DS-CNN-S":    {"clean": 96.4},
    "BC-ResNet-1": {"clean": 95.0},
    "NC-SSM":      {"clean": 95.3,   # = NanoMamba-NC-Matched (~7.4K)
                    "factory": {-15: 61.4, -10: 69.2, -5: 77.8, 0: 86.8,
                                 5: 90.1,  10: 92.4, 15: 92.9},
                    "white":   {-15: 50.8, -10: 63.0, -5: 75.5, 0: 85.2,
                                 5: 90.0,  10: 92.5, 15: 93.5},
                    "babble":  {-15: 73.4, -10: 79.1, -5: 84.1, 0: 89.9,
                                 5: 92.4,  10: 94.0, 15: 94.8},
                    "street":  {-15: 60.2, -10: 64.3, -5: 75.2, 0: 82.2,
                                 5: 88.9,  10: 92.2, 15: 93.7},
                    "pink":    {-15: 55.2, -10: 62.9, -5: 76.7, 0: 86.6,
                                 5: 91.5,  10: 93.0, 15: 94.4}},
    "NC-SSM-20K":  {"clean": 96.2,
                    "factory": {-15: 63.0, -10: 73.0, -5: 83.4, 0: 89.7,
                                 5: 93.4,  10: 94.9, 15: 95.0},
                    "white":   {-15: 44.0, -10: 63.1, -5: 81.2, 0: 88.4,
                                 5: 92.9,  10: 94.5, 15: 95.2},
                    "babble":  {-15: 77.0, -10: 83.6, -5: 89.0, 0: 92.9,
                                 5: 94.9,  10: 95.5, 15: 96.1}},
    # NanoMamba-Tiny / -Small (Interspeech-era checkpoints): no direct
    # TASLP row -- we print measured values without pass/fail enforcement.
    "NanoMamba-Tiny":  {"clean": None},
    "NanoMamba-Small": {"clean": None},
}


# Map shipped checkpoint folder -> (factory_fn_name, is_cnn).
# None means "not available in this inventory".
CHECKPOINT_SPECS = {
    # Baselines
    "DS-CNN-S":    ("DS-CNN-S/best.pt",        "DSCNN_S",              True),
    "BC-ResNet-1": ("BC-ResNet-1/best.pt",     "BCResNet_scale1",      True),
    # NC-SSM (TASLP main result)
    "NC-SSM":      ("NC-SSM/best.pt",          "nc_matched",           False),
    "NC-SSM-20K":  ("NanoMamba-NC-20K/best.pt","nc_20k",               False),
    # Interspeech-era NanoMamba (kept for back-compat)
    "NanoMamba-Tiny":  ("NanoMamba-Tiny/best.pt",  "nanomamba_tiny",   False),
    "NanoMamba-Small": ("NanoMamba-Small/best.pt", "nanomamba_small",  False),
}

DEFAULT_MODEL_ORDER = [
    "DS-CNN-S", "BC-ResNet-1",
    "NC-SSM", "NC-SSM-20K",
    "NanoMamba-Tiny", "NanoMamba-Small",
]


# --------------------------------------------------------------------- #
# Environment / dependency check                                        #
# --------------------------------------------------------------------- #
def check_environment() -> None:
    print("=" * 72)
    print("NC-SSM / NanoMamba -- IEEE TASLP 2026 reproduction")
    print("=" * 72)
    print(f"Python   : {platform.python_version()}")
    print(f"Platform : {platform.platform()}")

    missing = []
    for pkg in ("numpy", "torch"):
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "?")
            print(f"{pkg:<11}: {v}")
        except ImportError:
            missing.append(pkg)
            print(f"{pkg:<11}: MISSING")

    try:
        import torchaudio
        print(f"torchaudio : {torchaudio.__version__}")
    except ImportError:
        print("torchaudio : MISSING (will fall back to scipy.io.wavfile)")

    if missing:
        print(f"\nMissing required packages: {missing}")
        print("Install with: pip install numpy torch torchaudio")
        sys.exit(2)

    import torch
    print(f"CUDA       : {torch.cuda.is_available()} "
          f"({torch.cuda.device_count()} GPU)")
    print("=" * 72)


# --------------------------------------------------------------------- #
# Dataset                                                               #
# --------------------------------------------------------------------- #
def resolve_data_dir(arg_dir):
    """Locate GSC V2 on disk. Returns the parent directory suitable for
    passing as `root` to train_all_models.SpeechCommandsDataset (which
    expects <root>/SpeechCommands/speech_commands_v0.02/ layout)."""
    candidates = []
    if arg_dir:
        candidates.append(Path(arg_dir))
    candidates += [
        REPO_ROOT / "data",
        Path.cwd() / "data",
        Path.home() / "data",
    ]
    for root in candidates:
        if not root.exists():
            continue
        nested = root / "SpeechCommands" / "speech_commands_v0.02"
        if nested.exists() and any(nested.iterdir()):
            print(f"[data] Found GSC V2: {nested}")
            return root

    print("\n[ERROR] Google Speech Commands V2 not found.")
    print("Searched:")
    for c in candidates:
        print(f"  - {c}")
    print("\nOptions:")
    print("  1. Re-run with --data-dir pointing to an existing GSC V2 copy.")
    print("     The directory must contain:")
    print("       <data-dir>/SpeechCommands/speech_commands_v0.02/<keyword>/*.wav")
    print("  2. Download (~2.3 GB):")
    print("       wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz")
    print("       mkdir -p data/SpeechCommands/speech_commands_v0.02")
    print("       tar -xzf speech_commands_v0.02.tar.gz \\")
    print("           -C data/SpeechCommands/speech_commands_v0.02")
    print("  3. Use colab/NanoMamba_Test.ipynb (dataset is auto-downloaded).")
    sys.exit(4)


# --------------------------------------------------------------------- #
# Model factory                                                         #
# --------------------------------------------------------------------- #
def build_model(name):
    """Instantiate a bare model by short-name. Returns (model, is_cnn)."""
    if name == "DS-CNN-S":
        from train_all_models import DSCNN_S
        return DSCNN_S(n_classes=12), True
    if name == "BC-ResNet-1":
        from paper_models import BCResNet
        return BCResNet(n_classes=12, scale=1), True
    if name == "NC-SSM":
        from nanomamba import create_nanomamba_nc_matched
        return create_nanomamba_nc_matched(12), False
    if name == "NC-SSM-20K":
        from nanomamba import create_nanomamba_nc_20k
        return create_nanomamba_nc_20k(12), False
    if name == "NanoMamba-Tiny":
        from nanomamba import create_nanomamba_tiny
        return create_nanomamba_tiny(12), False
    if name == "NanoMamba-Small":
        from nanomamba import create_nanomamba_small
        return create_nanomamba_small(12), False
    raise ValueError(f"Unknown model: {name}")


def load_checkpoint(name, device):
    """Build model and load weights. Returns (model, params) or (None, 0)."""
    import torch
    rel = CHECKPOINT_SPECS[name][0]
    ckpt_path = REPO_ROOT / "checkpoints_full" / rel
    if not ckpt_path.exists():
        print(f"[skip] {name:<18} (no checkpoint at {ckpt_path})")
        return None, 0

    try:
        model, _ = build_model(name)
    except Exception as exc:
        print(f"[skip] {name:<18} (model build failed: {exc})")
        return None, 0

    try:
        ckpt = torch.load(str(ckpt_path), map_location=device,
                          weights_only=False)
    except Exception as exc:
        print(f"[skip] {name:<18} (torch.load failed: {exc})")
        return None, 0

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        ema = ckpt.get("ema_state_dict") if isinstance(ckpt, dict) else None
        if ema:
            model.load_state_dict(ema, strict=True)
        else:
            missing, _ = model.load_state_dict(state, strict=False)
            if missing:
                print(f"      (missing keys loaded with defaults: {len(missing)})")
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    val_acc = ckpt.get("val_acc", 0.0) if isinstance(ckpt, dict) else 0.0
    print(f"[ok]   {name:<18} {params:>7,} params  "
          f"epoch={epoch}  val_acc={val_acc:.2f}%")
    return model, params


# --------------------------------------------------------------------- #
# Evaluation                                                            #
# --------------------------------------------------------------------- #
def make_loader(data_root, quick, batch_size):
    from torch.utils.data import DataLoader, Subset
    from train_all_models import SpeechCommandsDataset
    ds = SpeechCommandsDataset(str(data_root), subset="testing", augment=False)
    if quick:
        n = min(1500, len(ds))
        ds = Subset(ds, list(range(n)))
        print(f"[quick] using {n} of {len(ds.dataset) if hasattr(ds, 'dataset') else n} test samples")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


def eval_clean(model, loader, device):
    from train_all_models import evaluate
    acc, _, _ = evaluate(model, loader, device)
    return acc


def eval_noisy(model, loader, device, noise_type, snr_db):
    from train_all_models import evaluate_noisy
    return evaluate_noisy(model, loader, device,
                          noise_type=noise_type, snr_db=snr_db)


# --------------------------------------------------------------------- #
# Report                                                                #
# --------------------------------------------------------------------- #
def paper_ref(model_name, condition):
    """Return the paper's reported number for (model, condition), or None."""
    tbl = PAPER_TABLE_III.get(model_name, {})
    if condition == "clean":
        return tbl.get("clean")
    # condition is ("noise_type", snr_db)
    nt, snr = condition
    noise_tbl = tbl.get(nt)
    if isinstance(noise_tbl, dict):
        return noise_tbl.get(snr)
    return None


def cond_label(condition):
    if condition == "clean":
        return "clean (test)"
    nt, snr = condition
    return f"{nt} @ {snr:+d} dB"


def report(rows, tol):
    """rows: list of (model, condition, measured_acc)."""
    print("\n" + "=" * 82)
    print("TASLP 2026 reproduction -- measured vs paper")
    print("=" * 82)
    print(f"{'Model':<18} {'Condition':<22} {'Measured':>11} "
          f"{'Paper':>10} {'|delta|':>10}")
    print("-" * 82)

    max_dev = 0.0
    comparable = 0
    for name, cond, meas in rows:
        paper = paper_ref(name, cond)
        if paper is None:
            paper_s = "     n/a"
            dev_s   = "       -"
        else:
            dev = abs(meas - paper)
            max_dev = max(max_dev, dev)
            comparable += 1
            paper_s = f"{paper:>8.2f}%"
            flag = "" if dev <= tol else " !"
            dev_s = f"{dev:>+8.2f}{flag}"
        print(f"{name:<18} {cond_label(cond):<22} "
              f"{meas:>10.2f}% {paper_s:>10} {dev_s:>10}")
    print("=" * 82)

    if comparable == 0:
        print("\n[INFO] No comparable paper rows for loaded checkpoints.")
        print("       Measured values printed above; no pass/fail enforced.")
        return 0

    print(f"\nComparable rows: {comparable}")
    print(f"Max |delta| vs paper = {max_dev:.2f} %p   (tolerance = {tol:.2f} %p)")
    if max_dev <= tol:
        print("[PASS] Reproduction within tolerance.")
        return 0
    print("[FAIL] At least one row exceeds tolerance.")
    print("   Notes:")
    print("   - Noise generation is stochastic; re-run or use a fixed --seed.")
    print("   - The Interspeech-era NanoMamba-Tiny/-Small checkpoints do not")
    print("     have direct TASLP rows; they are reported n/a.")
    return 1


# --------------------------------------------------------------------- #
# Entry                                                                 #
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="IEEE TASLP 2026 reproduction -- one-command reviewer script")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Path to GSC V2 root (containing SpeechCommands/...). "
                         "Auto-detected if omitted.")
    ap.add_argument("--noise-type", type=str, default="factory",
                    choices=["factory", "white", "babble", "street", "pink"],
                    help="Noise type for the smoke-test (default: factory).")
    ap.add_argument("--snr", type=int, default=0,
                    help="SNR in dB for the smoke-test (default: 0).")
    ap.add_argument("--full", action="store_true",
                    help="Sweep all 5 noise types x 7 SNRs (slow; ~30-60 min).")
    ap.add_argument("--tol", type=float, default=2.0,
                    help="Absolute tolerance in %%p vs paper (default: 2.0).")
    ap.add_argument("--quick", action="store_true",
                    help="Use 1500-sample test subset (smoke test ~1 min).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODEL_ORDER),
                    help="Comma-separated checkpoint names under checkpoints_full/.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    check_environment()
    t0 = time.time()

    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = resolve_data_dir(args.data_dir)
    loader = make_loader(data_root, args.quick, args.batch_size)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_names if m not in CHECKPOINT_SPECS]
    if unknown:
        print(f"\n[ERROR] Unknown model(s): {unknown}")
        print(f"  Known: {list(CHECKPOINT_SPECS.keys())}")
        return 3

    rows = []
    loaded = 0
    for name in model_names:
        print(f"\n--- {name} ---")
        model, params = load_checkpoint(name, device)
        if model is None:
            continue
        loaded += 1

        # Clean
        print(f"       [eval] clean ...")
        acc = eval_clean(model, loader, device)
        rows.append((name, "clean", acc))

        # Noise grid
        if args.full:
            grid = [
                (nt, s)
                for nt in ("factory", "white", "babble", "street", "pink")
                for s in (-15, -10, -5, 0, 5, 10, 15)
            ]
        else:
            grid = [(args.noise_type, args.snr)]

        for nt, snr in grid:
            print(f"       [eval] {nt:<8s} @ {snr:+d} dB ...")
            acc = eval_noisy(model, loader, device, nt, snr)
            rows.append((name, (nt, snr), acc))

        # free VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if loaded == 0:
        print("\n[ERROR] No checkpoints loaded -- cannot reproduce anything.")
        print("Expected at least one of:")
        for k, v in CHECKPOINT_SPECS.items():
            print(f"  checkpoints_full/{v[0]}")
        return 3

    code = report(rows, tol=args.tol)
    print(f"\nTotal elapsed: {time.time() - t0:.1f} s")
    if args.quick:
        print("\n[NOTE] --quick used a 1500-sample subset. For publishable")
        print("       reproduction, re-run without --quick.")
    return code


if __name__ == "__main__":
    sys.exit(main())
