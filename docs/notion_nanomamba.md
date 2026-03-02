# NanoMamba

## 4.8KB AI that hears through noise.

> 세상에서 가장 작은 음성 인식 엔진.
> 소음 속에서도 듣고, 배터리는 1년 갑니다.

---

# The Numbers

> **4.8 KB**
> 모델 전체 크기. 문자 메시지 1개보다 작습니다.

> **0.94 ms**
> 응답 시간. 눈 깜빡이는 속도의 1/300.

> **288 Days**
> CR2032 코인 배터리 하나로. 충전 없이.

> **10x**
> 동급 CNN 대비 연산량 절감. 같은 일을 10배 적은 에너지로.

---

# Why NanoMamba?

## The Problem

공장에서 작업자가 **"멈춰!"** 라고 외칩니다.
하지만 기계 소음이 너무 커서, AI는 듣지 못합니다.

기존 음성 인식(KWS) 기술의 현실:

- 조용한 환경에서는 96% 정확도
- **공장 소음(-15dB)에서는 28%로 추락** — 랜덤 추측 수준
- 소음 제거 칩을 별도로 추가? → 전력 2배, 비용 증가, BOM 복잡

## Our Answer

NanoMamba는 **소음을 이해하는 AI**입니다.

별도 노이즈 캔슬링 없이, 모델 자체가 소음 환경에 적응합니다:

- **시끄러운 구간** → 긴 기억력으로 소음을 걸러냄
- **조용한 구간** → 빠른 반응으로 음성을 즉시 전달
- **노이즈 유형 자동 판별** → 공장 소음과 사람 소리를 구분하여 처리 전략 변경

추가 하드웨어 **제로**. 추가 파라미터 **제로**. 추가 비용 **제로**.

---

# Impact: NanoMamba vs CNN Baselines

같은 파라미터 수(~7,400개)에서 비교했습니다.

## 정확도 (Parameter-Matched)

> **NanoMamba-Matched: 95.1%** vs BC-ResNet-1: 95.3%
> 단 0.2%p 차이. 사실상 동등한 성능.

## 연산 효율 (같은 정확도에서)

| | NanoMamba-Matched | BC-ResNet-1 | 차이 |
|---|---|---|---|
| Clean 정확도 | **95.1%** | 95.3% | 0.2%p (동등) |
| 연산량 (MACs) | **0.68M** | 4.62M | **6.8x 절감** |
| 응답 시간 (Cortex-M7) | **1.42 ms** | 9.63 ms | **6.8x 빠름** |
| 메모리 사용량 | **31.7 KB** | 102.0 KB | **3.2x 절감** |
| 배터리 수명 (CR2032) | **212일** | 31일 | **6.8x 오래** |

## 소음 강건성

> NanoMamba + SS+Bypass (0-파라미터 전처리) 적용 시,
> 극심한 소음(-15dB)에서 **백색소음 +23.8%p, 핑크소음 +29.2%p 개선**.
> 깨끗한 환경 정확도는 **100% 보존**.

| 소음 유형 | NanoMamba 개선 | BC-ResNet-1 개선 | 승자 |
|---|---|---|---|
| White (-15dB) | **+23.8%p** | +0.5%p | NanoMamba |
| Pink (-15dB) | **+29.2%p** | +4.4%p | NanoMamba |
| Street (-15dB) | **+1.7%p** | +0.5%p | NanoMamba |
| Factory (-15dB) | **+3.6%p** | +6.2%p | BC-ResNet-1 |
| Babble (-15dB) | -0.1%p | +0.3%p | 동등 |

**CNN은 소음 제거 신호를 잘 활용하지 못합니다. NanoMamba는 다릅니다.**

---

# Use Cases

## Hearing Aids & Hearables

> 보청기 배터리가 9배 오래 갑니다.

- CR2032 코인셀로 288일 연속 동작
- 소음 환경(식당, 거리) 강건성 내장
- Nordic nRF54 BLE 오디오 호환
- 노이즈 캔슬링 칩 불필요 → BOM 절감

## Industrial IoT

> 공장 소음 속에서도 음성 명령을 인식합니다.

- -15dB SNR 환경 동작 검증 완료
- 가장 작은 MCU에서도 동작 (23.8KB RAM)
- 실시간 응답 (0.94ms)
- 별도 전처리 HW 없이 소음 적응

## Smart Home

> 배터리 교체 없이 1년 가는 음성 센서.

- 초인종, 리모컨, 침대 센서 등
- Always-on 키워드 감지
- 코인셀 하나로 연단위 동작

## Wearables

> 시계 위의 AI. 0.94ms 응답.

- 워치/밴드 폼팩터 호환
- 23.8KB — 어떤 Cortex-M에서도 동작
- Sub-millisecond 반응속도

---

# ARM Deployment

모든 ARM Cortex-M 시리즈에서 즉시 배포 가능합니다.

| 프로세서 | NanoMamba 지연시간 | 배터리 수명 (CR2032) |
|---|---|---|
| Cortex-M4 (168MHz) | 2.67 ms | ~200일 |
| Cortex-M7 (480MHz) | 0.94 ms | 143일 |
| Cortex-M33 (128MHz) | 3.51 ms | **288일** |
| Cortex-M55+Ethos (250MHz) | **0.22 ms** | **1,680일** (4.6년) |

> 모델 크기 4.8KB — INT8 양자화 시 **어떤 MCU의 Flash에도** 들어갑니다.

---

# Technology

NanoMamba의 핵심 기술은 3가지입니다.

## Spectral-Aware SSM (SA-SSM)

주파수 대역별 SNR을 실시간 추정하여, SSM의 시간 역학을 동적 제어합니다.

- 고 SNR → 빠른 적응 (음성 전달)
- 저 SNR → 긴 기억 (소음 억제)
- 기존 SSM/Mamba와 다른 점: 선택 파라미터(dt, B)가 SNR에 직접 연동

## DualPCEN: 이중 전문가 노이즈 라우팅

2개의 PCEN 전문가가 소음 유형에 따라 자동 전환됩니다.

- Expert 1: 비정상 소음 전문 (사람 목소리, babble)
- Expert 2: 정상 소음 전문 (공장, 백색소음)
- 라우팅 기준: Spectral Flatness (학습 파라미터 0개)

## SS+Bypass: 0-파라미터 소음 제거

Spectral Subtraction + SNR-Adaptive Bypass.

- 추가 학습 파라미터 없는 클래식 신호처리
- SNR 높으면 → 원본 그대로 통과 (bypass)
- SNR 낮으면 → 스펙트럴 차감 적용
- 깨끗한 환경 정확도 손실 제로

---

# Model Family

용도에 따라 3가지 모델을 선택할 수 있습니다.

| 모델 | 파라미터 | 크기 (INT8) | Clean Acc | 용도 | 상태 |
|---|---|---|---|---|---|
| **Tiny** | 4,957 | 4.8 KB | 93.7% | 극소형 IoT, 보청기 | Ready |
| **Matched** | 7,402 | 7.2 KB | **95.1%** | BC-ResNet-1 대체 | **Ready** |
| **Small** | 12,355 | 12.1 KB | - | 고정밀 애플리케이션 | Available |

> 모든 모델이 DualPCEN + SA-SSM + SS+Bypass를 공유합니다.
> 동일 아키텍처, 스케일만 다릅니다.

---

# IP & Publication

- **Journal**: IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2026
- **Patent**: 한국 특허 출원 완료 (SA-SSM + DualPCEN + MoE Routing)
- **License**: 학술/연구 무료. 상업 라이선스 별도 협의.

---

# Contact

## SmartEar Co., Ltd.

기술 파트너십, 라이선스, 공동 개발 문의:

**Jin Ho Choi, Ph.D.**
jinhochoi@smartear.co.kr

---

> *NanoMamba — When every microjoule counts and every word matters.*
