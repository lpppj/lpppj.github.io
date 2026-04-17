---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2024](https://openreview.net/pdf?id=lJkOCMP2aW)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting (ICLR 2024)

## Abstract

- 기존 Transformer for TS 모델들은 limited or fixed scales
- Pathformer는 temporal resolution과 temporal distance를 합치는 방식
  - 다양한 크기의 patches를 사용해서 서로 다른 temporal resolutions를 만들고
  - dual attention으로 temporal dependencies(global correlation과 local details)
- Input의 varying temporal dynamics에 따라 adaptive multi-scale

## 1. Introduction

- 최근 simpler linear model의 성능이 transformer 보다 잘 나옴 - transformer 디자인 수정 필요
- Time Series에서는 multiple scales를 고려하는 것이 중요 (일, 월 등...)
  - **Temporal resolution**: patch의 길이 (fine-grained or coarse grained)
  - **Temporal distance** : time steps 사이의 거리
![그림1](/assets/img/timeseries/Pathformer/fig1.png)
  - 그림 왼쪽 위에서 blue patch와 orange patch는 Temporal resolution가 다른 것이고
  - 그림 오른쪽에서 black arrows와 colored arrows는 Temporal distance가 다른 것

- 그래서 Transformer로 multi-scale modeling 하려는데 challenges 2개
  - **incompleteness of multi-scale modeling**
    - 단순히 Temporal resolution (patch 길이) 늘리는 건 다양한 dependency 파악에 도움 안됨
    - 차라리 Temporal distance (time steps 간격) 다양하게 하면 좋은데 Temporal distance는 patch size (data division)에 따라 달라짐
  - **fixed multi-scale modeling process**
    - different series prefer different scales !
    - 그림 왼쪽 위 series는 rapid fluctuation, fine-grained, short-term characteriestics
    - 그림 왼쪽 아래 series는 coarse-grained and long-term characteriestics
    - 모든 데이터에 fixed multi-scale modeling하면 안되는데 매번 optimal scale 찾으려니 오래걸림
  - 그래서 **adaptive multi-scale modeling**이 필요하다.
    - adaptively models the current data from certain multiple scales

- 기존에는 그냥 다양한 길이의 patch로 자른 뒤에, patch 안에서 temporal dependencies를, patch 끼리 global correlation을 파악하는 dual attention
- 본 논문에서 제시하는 Pathformer는
  - Multi-scale router가 seasonality와 trend를 보고 적절한 patch size들을 결정
  - 그 다음 multi-scale characteristics를 weighted aggregation

## 2. Related Work

- Time Series forecasting
  - Statistical modeling, GNN(spatial dependency), RNN(temporal dependency), CNN(sub-series features), TimesNet(1-dim $$\to$$ 2-dim), LLM-based...
- Transformer
  - Informer : prob-sparse self-attention to select important keys
  - Triformer :  manages to reduce the complexity
  - Autoformer : auto-correlation mechanisms to replace self- attention
  - FEDformer : the perspective of frequency to model temporal dynamics
  - 하지만 simple linear model의 성능이 더 좋은 경우가 많았음
  - PatchTST : patching and channel independence로 transformer의 가능성 제시
- Multi-scale Modeling for Time Series
  - N-HiTs : multi-rate data sampling and hierarchical interpolation for diverse temporal resolutions
  - Pyraformer : pyramid attention to extract features at different temporal resolutions
  - 하지만 fixed scale이었고, 본 논문에서는 adaptive multi scale 제안

## 3. Methodology

![그림2](/assets/img/timeseries/Pathformer/fig2.png)

- Instance Norm - Stacking of **Adaptive Multi-Scale Blocks** - Predictor(FC)로 구성
- Adaptive Multi-Scale Blocks은 multi-scale Transformer block과 adaptive pathways으로 구성
  - multi-scale Transformer block
    - 다양한 size의 patch division and dual attention으로 multi-scale temporal resolution and distances 통합
  - adaptive pathways
    - multi-scale router가 다양한 patch size르 고르고
    - aggregator에서 mutli-scale characteristics를 weighted aggregation

### 3.1. Multi-scale Transformer Block

- **Multi-scale Division**
  - $$\mathrm{X} \in \mathbb{R}^{H \times d}$$를 P개의 patch $$\left(\mathrm{X}^1, \mathrm{X}^2, \ldots, \mathrm{X}^P\right)$$로 나눔, patch size는 M개 $$\mathcal{S}=\left\{S_1, \ldots, S_M\right\}$$ 즉 $$P = H/S$$
  - 다양한 크기의 patch size로 다양한 temporal resolutions를 dual attention

- **Dual attention**
![그림3](/assets/img/timeseries/Pathformer/fig3.png)
  - Intra-patch attention : relationships btw time steps within each patch
    - 먼저 $$X_{\text {intra }}^i \in \mathbb{R}^{S \times d_m}$$ patch를 embedding하고 $$\operatorname{Attn}_{\text {intra }}^i=\operatorname{Softmax}\left(Q_{\text {intra }}^i\left(K_{\text {intra }}^i\right)^T / \sqrt{d_m}\right) V_{\text {intra }}^i \in \mathbb{R}^{1 \times d_m}$$
    - 모든 patches에 대해 다 합치면 $$\operatorname{Attn}_{\text {intra }}=\operatorname{Concat}\left(\operatorname{Attn}_{\text {intra }}^1, \ldots, \operatorname{Attn}_{\text {intra }}^P\right) \in \mathbb{R}^{P \times d_m}$$​
    - Linear transformation으로 $$\operatorname{Attn}_{\text {intra }} \in \mathbb{R}^{P \times S \times d_m}$$
  - Inter-patch attention : relationships btw patches to capture global correlations
    - 먼저 feature embedding and rearrange : $$\mathrm{X}_{\text {inter }} \in \mathbb{R}^{P \times d_m^{\prime}} \text {, where } d_m^{\prime}=S \cdot d_m$$
    - 아까처럼  linear mapping으로 $$\operatorname{Attn}_{\text {inter }}=\operatorname{Softmax}\left(Q_{\text {inter }}\left(K_{\text {inter }}\right)^T / \sqrt{d_m^{\prime}}\right) V_{\text {inter }} \in \mathbb{R}^{P \times S \times d_m}$$
  - 둘을 더하면 $$\mathrm{Attn}_{\text {intra }} + \mathrm{Attn}_{\text {intra }}=\text { Attn } \in \mathbb{R}^{P \times S \times d_m}$$

### 3.2. Adaptive Pathways

-  different series may prefer diverse scales $$\to$$ model needs to figure out critical scales based on the input
-  **Multi-scale router** selects specific sizes of patch division based on the input data
-  **Multi-scale aggregator** combines multi-scale characteristics(weighted aggregation) $$\to$$ output of the Transformer block

- **Multi-scale router** : selects the optimal sizes for patch division
  - by its complex inherent characteristics and dynamic patterns
  - seasonality and trend decomposition $$\to$$ extract periodicity and trend patterns
  - **Seasonality decomposition**
    - Discern Fourier Transform (DFT) : X를 푸리에 basis로 decomopose하고 the largest amplitudes 선택 (to keep the sparsity of frequency domain)
    - IDFT로 periodic patterns $$\mathrm{X}_{\text {sea }}=\operatorname{IDFT}\left(\left\{f_1, \ldots, f_{K_f}\right\}, A, \Phi\right)$$ 얻음
      - $$\Phi$$, $$A$$ :the phase and amplitude of each frequency from $$\operatorname{DFT}(\mathrm{X})$$
      - $$\left\{f_1, \ldots, f_{K_f}\right\}$$ : the frequencies with the top $$K_f$$ amplitudes
  - **Trend decomposition**
    - seasonality를 제외한 부분 $$\mathrm{X}_{\mathrm{rem}}=\mathrm{X}-\mathrm{X}_{\text {sea }}$$에 대해 different kernels of average pooling for moving averages to extract trend patterns
    - 정리하면 $$\mathrm{X}_{\text {trend }}=\operatorname{Softmax}\left(L\left(\mathrm{X}_{\mathrm{rem}}\right)\right) \cdot\left(\operatorname{Avgpool}\left(\mathrm{X}_{\mathrm{rem}}\right)_{\text {kernel }_1}, \ldots, \operatorname{Avgpool}\left(\mathrm{X}_{\mathrm{rem}}\right)_{\text {kernel }_N}\right)$$
  - input $$X$$에 seasonality pattern and trend pattern를 더하고 linear mapping으로 $$\mathrm{X}_{\text {trans }} \in \mathbb{R}^d$$
  - 마지막으로 routing function $$R\left(\mathrm{X}_{\text {trans }}\right)=\operatorname{Softmax}\left(\mathrm{X}_{\text {trans }} W_r+\epsilon \cdot \operatorname{Softplus}\left(\mathrm{X}_{\text {trans }} W_{\text {noise }}\right)\right), \epsilon \sim \mathcal{N}(0,1)$$ 을 통해 pathway weights $$\in \mathbb{R}^{M}$$을 구한다.
    - $$W_r \text { and } W_{\text {noise }} \in \mathbb{R}^{d \times M}$$은 leanable parameters, $$d$$는 feature dim, $$M$$은 patch size의 개수
    - noise를 추가한 이유는 patch size가 같은 것만 나오는 것을 방지하기 위함
    - 이렇게 만든 M개의 pathway weights 중에서 top K개를 제외하고 0으로 만든 걸 $$\bar{R}\left(\mathrm{X}_{\text {trans }}\right)$$로 표기
- **Multi-Scale Aggregator**
  - $$\bar{R}\left(\mathrm{X}_{\text {trans }}\right)_i>0$$이 의미하는 것은 patch size $$S_i$$로 나누고 dual attention을 수행함을 의미
  - 그러므로 AMS block의 final output은 $$\mathrm{X}_{\text {out }}=\sum_{i=1}^M \mathcal{I}\left(\bar{R}\left(\mathrm{X}_{\text {trans }}\right)_i>0\right) R\left(\mathrm{X}_{\text {trans }}\right)_i T_i\left(\mathrm{X}_{\text {out }}^i\right)$$
    - $$\mathrm{X}_{\text {out }}^i$$는 output of the multi-scale Transformer with the patch size $$S_i$$
    - $$T_i$$는 align the temporal dimension from different scales하는 transformation function

## 4. Experiments

![그림4](/assets/img/timeseries/Pathformer/table1.png)

![그림5](/assets/img/timeseries/Pathformer/table3.png)

## 5. Conclusion
- Pathformer : Multi-Scale Transformer with Adaptive Pathways for TSF
  - integrates multi-scale temporal resolutions and temporal distances
  - by patch division with multiple patch sizes and dual attention (modeling multi-scale characteristics)
  - adaptive pathways dynamically select and aggregate scale-specific characteristics