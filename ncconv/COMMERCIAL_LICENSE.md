# Commercial License -- NC-Conv-SSM Vision (`ncconv/`)

This document describes commercial licensing terms for the Vision source code
in the `ncconv/` directory (Noise-Conditioned Convolution + temporal NC-SSM,
targets: ICCV 2027, BMVC 2026, ACCV 2026, CVPR 2027, IEEE TIP 2027).
For the academic (non-commercial) license, see [LICENSE](LICENSE) in this
directory.

---

## 1. When a Commercial License is Required

You MUST obtain a Commercial License before any of the following activities:

| Activity | Requires Commercial License |
|---|---|
| Academic research, publication, thesis | No (Academic License covers) |
| Teaching, classroom demonstration | No |
| Reproducing benchmarks published in the paper | No |
| Incorporation into a commercial product / SaaS | Yes |
| Internal use at a for-profit company (production) | Yes |
| Automotive ADAS / autonomous driving deployment | Yes |
| Drone, robotics, or surveillance deployment | Yes |
| MCU / NPU / FPGA / ASIC implementation for sale | Yes |
| Redistribution inside a commercial bundle | Yes |
| Commercialization of downstream academic work | Yes |

**Note on academic-to-commercial path.** If your project starts under the
Academic License (thesis, grant-funded research, open-source exploration)
and later moves toward commercialization -- spin-off, licensing to industry,
product development, revenue-generating deployment -- a Commercial License
must be signed **before** the commercialization step begins.

---

## 2. License Tiers (Framework)

Final pricing is negotiated case-by-case. Indicative tiers:

### Tier A -- Evaluation / Pilot (6 months)
- Single team, single product, single deployment site.
- No redistribution, no sub-licensing.
- Indicative fee: low five-figure USD range.

### Tier B -- Product Integration (annual)
- Integration into one commercial product line (e.g., one ADAS SKU,
  one drone model, one surveillance appliance).
- Redistribution only as part of the licensed product binary.
- Royalty or flat annual fee, negotiated.

### Tier C -- Enterprise / OEM
- Multi-product, multi-site, redistribution rights.
- Optional sublicensing to downstream OEMs / Tier-1 suppliers.
- Custom terms.

### Tier D -- Defense / Government
- Handled under separate terms including export-control compliance
  (ITAR / Wassenaar / Korean DAPA regulations as applicable).

### Academic Spin-off Clause
- Reduced-rate commercial license available for university spin-offs
  within the first 24 months, subject to equity or revenue-share terms.

---

## 3. Patent Scope

A Commercial License includes a field-of-use patent grant covering the
pending patents listed below, limited to the licensed product scope:

- **NanoMamba / NC-SSM** -- Noise-Conditioned State Space Models
  (KR + US pending).
- **NC-Conv (spatial)** -- Dual-path static/dynamic convolution with a
  learned per-sample or per-spatial quality gate sigma (KR pending; US in
  preparation).
- **Temporal NC-SSM for video** -- Bidirectional temporal NC-SSM with
  per-frame quality gate for tunnel / adverse-weather / illumination-shock
  scenarios (KR pending).

The Academic License grants **no** commercial patent rights.

---

## 4. Warranty, Support, and Indemnification

- **As-is under Academic License** (no support, no warranty).
- **Commercial Licenses** may include support SLA, maintenance updates,
  and mutual indemnification as negotiated.

---

## 5. Contact

All commercial license inquiries:

> **Dr. Jin Ho Choi** -- SmartEAR, Daegu, Republic of Korea
> Email: **jinhochoi@smartear.co.kr**
> Subject line: `[NC-Conv-SSM Commercial License] <your organization>`

Please include in your first email:
1. Organization name and country of incorporation.
2. Intended use case (product, internal tool, automotive, drone,
   surveillance, defense, research, etc.).
3. Deployment scale (units / vehicles / sites).
4. Timeline (evaluation start, production launch).
5. Whether hardware (MCU / NPU / FPGA / ASIC) is involved.

A written NDA can be executed before detailed technical discussion.

---

## 6. Attribution (Required Under Both Licenses)

Any academic publication, technical report, or product documentation that
uses this software must cite at least one of:

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

@inproceedings{choi2026ncconv_bmvc,
  author    = {Jin Ho Choi},
  title     = {NC-Conv: Noise-Conditioned Dual-Path Convolution},
  booktitle = {Proc. British Machine Vision Conference (BMVC)},
  year      = {2026},
  note      = {submitted}
}

@inproceedings{choi2026ncconv_accv,
  author    = {Jin Ho Choi},
  title     = {NC-Conv for Degraded Vision},
  booktitle = {Proc. Asian Conference on Computer Vision (ACCV)},
  year      = {2026},
  note      = {submitted}
}

@inproceedings{choi2027ncconv_cvpr,
  author    = {Jin Ho Choi},
  title     = {NC-Conv-SSM: Degradation-Robust Vision with Dual-Path
               Blending},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern
               Recognition (CVPR)},
  year      = {2027},
  note      = {submitted}
}
```

---

_Last updated: 2026-04-22_
