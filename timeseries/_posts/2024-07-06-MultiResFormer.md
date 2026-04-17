---
layout: post
related_posts:
  _
title: 
description: >
  [Arxiv 2023](https://arxiv.org/pdf/2311.18780)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# MultiResFormer: Transformer with Adaptive Multi-Resolution Modeling for General Time Series Forecasting (Arxiv 2023)

## Abstract

- Transformer-based models는 TS를 segments로 나눠서 encode (**patches**)
  - 다양한 **scale**(i.e.. **resolutions**)에서의 TS를 모델링
- 하지만 pre-defined scale(patch의 길이)은...
  - variety of intricate temporal dependencies를 찾기 어려움
- 그래서 MultiResFormer는 adaptive time scale.
  - patch length를 주어진 데이터의 주기성을 보고 찾겠다
  - 그 다음 intraperiod and interperiod dependencies 학습

## 1. Introduction

-  TSF를 위한 vanila Transformer의 modification 3:
  - 첫째. Efficient attention mechanisms for sub-quadratic attention computation
    - [LogTrans(2019)](https://arxiv.org/pdf/1907.00235), [Informer(2021)](https://arxiv.org/pdf/2012.07436), [Pyraformer(2022)](https://openreview.net/pdf?id=0EXmFzUn5I)
  - 둘째. Breaking the point-wise nature of dot-product attention for segment or series level dependency modeling
    - [LogTrans(2019)](https://arxiv.org/pdf/1907.00235), [Autoformer(2021)](https://arxiv.org/pdf/2106.13008), [Fedformer(2022)](https://arxiv.org/pdf/2201.12740), [PatchTST(2023)](https://arxiv.org/pdf/2211.14730), [Crossformer(2023)](https://openreview.net/pdf?id=vSVLM2j9eie)
  - 셋째. Modeling sequences at multiple time scales with hierarchical representation learning
    - [LogTrans(2019)](https://arxiv.org/pdf/1907.00235), [Triformer(2022)](https://arxiv.org/pdf/2204.13767), [Scaleformer(2023)](https://arxiv.org/pdf/2206.04038), [Pathformer(2024)](https://openreview.net/pdf?id=lJkOCMP2aW)
  - 하지만 세 번째 multi-scale methods의 경우 pre-defined resolution으로 인해 generalization이 안된다.
  - 그래서 본 논문에서는 데이터의 underlying periodicities을 파악해서 데이터에 맞게 multi-resolution view

![그림1](/assets/img/timeseries/MultiResFormer/fig1.png)

- 본 논문에서는 two core Transformer sublayers를 repurpose:
  - **Multi-headed attention** for "interperiod" variation modeling
    - MHA는 전역적인 패치 간 의존성을 모델링하는 데 강점을 가지고 있어 interperiod 변동을 모델링하는 데 적합
  - **Position-wise Feed-Forward network** for "intraperiod" variation modeling
    -  FFN은 각 위치 내의 복잡한 의존성을 모델링하는 데 강점을 가지고 있어 intraperiod 변동을 모델링하는 데 적합

- 고려해야 할 사항들
  - 어떻게 different resolution branches끼리 parameter-sharing을 할까 ?
    - 일단 parameter-sharing을 해야 특정 scale에 overfitting되는 걸 방지하는 건 맞음
    - patch length 모르니까 linear projection 못 쓰고, 그냥 padding하는 건 모델 학습을 방해함
    - 그래서 각 scale에서의 patches의 길이를 맞추기 위한 **interpolation scheme** 사용
    - 그리고 **resolution embedding**으로 scale-awareness
  - 계산 복잡도를 어떻게 줄일까 ?
    - 이미 interpolation scheme으로 patches의 길이를 맞춰줬으니 별도의 embedding 필요 없음
      - Dlinear model의 성공 사례에서 영감 받음

## 2. Related Work

### Time series Transformer

- Transformer의 quadratic complexity 때문에 longer series는 모델링하기 어려웠음
  - [LogTrans(2019)](https://arxiv.org/pdf/1907.00235) : sparse attention blocks where each token attends to others with an exponential step size
  - [Informer(2021)](https://arxiv.org/pdf/2012.07436) : entropy-based measurement to filter out uninformative keys for $$O(N)$$
  - [Triformer(2022)](https://arxiv.org/pdf/2204.13767), [Pyraformer(2022)](https://openreview.net/pdf?id=0EXmFzUn5I) : adopt CNN-like approaches for local attention operations
  - [PatchTST(2023)](https://arxiv.org/pdf/2211.14730) : patch단위로 attention 연산 하니까 $$O(N^2/S^2)$$

- 단일 시점의 데이터는 정보가 별로 없음 ([PatchTST(2023)](https://arxiv.org/pdf/2211.14730))
  - 단일 시점을 토큰으로 하는 transformer는 localized patterns을 간과할 수 있음
- channel-mixing embedding은 over-fitting 발생시킬 수 있음 ([PatchTST(2023)](https://arxiv.org/pdf/2211.14730))

### Multi-Resolution Time Series Modeling

- [TimesNet(2023)](https://arxiv.org/pdf/2210.02186)에서 adaptive multi-resolution modeling 하긴 함
  - 하지만 input length를 맞춰줘야 해서 flatten해야 하고 longer series 예측 못함
  - channel-mixing embedding이 불가피해서 overfitting

## 3. Adaptive Multi-Resolution Time Series Modeling with Transformers

- $$\mathbf{X}_{1 \ldots I}=\left(\mathbf{x}_1, \ldots, \mathbf{x}_I\right) \in \mathbb{R}^{I \times V}$$로 $$\mathbf{X}_{I+1 \ldots I+O}=\left(\mathbf{x}_{I+1}, \ldots, \mathbf{x}_{I+O}\right) \in \mathbb{R}^{O \times V}$$​ 예측

### 3.1. MultiResFormer

![그림2](/assets/img/timeseries/MultiResFormer/fig2.png)

- The periodicity-aware patching module, detecting salient periodicities (section 3.2.)
- The Transformer Encoder block, shared across all resolution branches (section 3.3.)
- aggregate the representations derived within each resolution branch into $$\mathbf{X}^{(l)} \in \mathbf{R}^{I \times V}$$​ (section 3.4.)

### 3.2. Salient Periodicity Detection

- salient periodicites of the input series는 Fast Fourier Transform (FFT)으로 찾음

  $$\begin{aligned}
  \mathbf{A} & =\operatorname{Avg}(\operatorname{Amp}(\operatorname{FFT}(\mathbf{X}))) \\
  \left\{f_1, \ldots, f_k\right\} & =\underset{f_* \in\left\{1, \ldots,\left\lfloor\frac{I}{2}\right\rfloor\right\}}{\operatorname{argTopk}}(\mathbf{A}) \\
  \text { Period }_i & =\left\lceil\frac{I}{f_i}\right\rceil
  \end{aligned}$$

- Gradient-based 방식이 아니라서 미분이 필요없음

### 3.3. Multi-Resolution Modeling with a Shared Transformer Block

- PatchTST처럼 fixed patch length 사용할 땐 high-dimensional embeddings을 위한 linear transformations가 필요했는데, 이제는 patch embedding layers가 필요없어짐 (efficient)
- 다양한 patch length를 사용한다고 해서 길이를 맞춰주기 위해 padding을 한다고 하더라도, MHA와 FFN에는 masking mechanism이 없기 때문에 모델 성능 저하
  - 그러므로 interpolation으로 original patches의 temporal characteristics 보존
- one resolution branch의 Periodi가 주어지면 길이 Periodi의 겹치치 않는 patch로 분할
  - 그 다음 길이 d가 되도록 linearly interpolate
  - shape of the patch-based representation of the input series : $$V \times\left\lceil\frac{I}{\text { Period }_i}\right\rceil \times d$$

- 각 resolution branch에서 same-sized patches로 연산이 이루어지기 때문에 resolution embedding을 linearly interpolate에 더해줌 (transformer block에게 해상도 알려주기 위해)
- MHA to capture patch-wise dependencies (interperiod variation modeling)
  - FFN layers for capturing dependencies within each patch (intraperiod variation modeling)

### 3.4. Adaptive Aggregation

- i 번째 resolution에서 representation의 shape은 $$V \times\left\lceil\frac{I}{\text { Period }_i}\right\rceil \times d$$​
- interpolation으로 $$I \times V$$로 representation
- $$I \times V$$으로 표현된 모든 k개의 resolution에서의 representation을 adaptive aggregation:

    $$\begin{aligned}
    & \left\{A_1, \ldots, A_k\right\}=\underset{f_* \in\left\{1, \ldots,\left\lfloor\frac{I}{2}\right\rfloor\right\}}{\operatorname{Topk}}(\mathbf{A}) \\
    & \left\{w_1, \ldots, w_k\right\}=\operatorname{Softmax}\left(\left\{A_1, \ldots, A_k\right\}\right)
    \end{aligned}$$

## 4. Experiments

- Main Results

![그림3](/assets/img/timeseries/MultiResFormer/table12.png)

- Ablation Study

![그림4](/assets/img/timeseries/MultiResFormer/table4.png)

- Varying Look-back Window Size

![그림5](/assets/img/timeseries/MultiResFormer/fig3.png)

- Representation analysis

![그림6](/assets/img/timeseries/MultiResFormer/fig4.png)

- Efficiency Comparison

![그림7](/assets/img/timeseries/MultiResFormer/fig5.png)

## 5. Conclusion

- 각 transformer blocks 내에서 FFT로 데이터의 underlying periodicities를 파악하고 resolution 결정
- 각 transformer blocks 내에서 interpolation 덕분에 resolution끼리 parameter sharing
- 블록 내의 encoder는 interpolation으로 input size와 output representation의 size가 같아서
  - embedding layer가 필요없고 final linear prediction head layer에서의 parameters 개수가 적음

  