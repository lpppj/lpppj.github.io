---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2024](https://openreview.net/pdf?id=7oLshfEIC2)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024)

## Abstract

- Real world에서 time series는 복잡한 temporal variations 때문에 forecasting이 어려움
- 지금까지는 그냥 decomposition하고 multiperiodicity 분석 했는데,
- 본 논문에서는 time series를 다양한 scale로 sampling하면 별개의 patterns가 관찰될 것이라는 intuition에 따라 multiscale-mixing을 제안
  - microscopic information은 fine scale, macroscopic information은 coarse scales

- **TimeMixer** : fully MLP-based architecture
  - with **Past-Decomposable-Mixing** (PDM) for past extraction
    - applies the decomposition to multiscale series
    - and mixes the decomposed seasonal and trend components
      - 즉 the microscopic seasonal과 macroscopic trend를 통합
  - with **Future-Multipredictor-Mixing** (FMM) for future prediction
    - multiscale observation을 통한 완성도 있는 예측을 위해 multiple predictors를 ensemble

## Introduction

- Time series의 representative models는 temporal variations를 파악
  - foundation backbone 별로 보면
  - CNN 계열로는 [MICN(2023)](https://openreview.net/forum?id=zt53IDUR1U), [TimesNet(2023)](https://arxiv.org/abs/2210.02186), [TCN(2020)](https://link.springer.com/article/10.1007/s00500-020-04954-0), RNN 계열로는 [LSTNet(2018)](https://arxiv.org/abs/1703.07015), [DA-RNN(2017)](https://arxiv.org/abs/1704.02971), [DeepAR(2020)](https://www.sciencedirect.com/science/article/pii/S0169207019301888)이 있고
  - Transformer 계열로는 [Informer(2021)](https://arxiv.org/abs/2012.07436), [Autoformer(2021)](https://arxiv.org/abs/2106.13008), [Fedformer(2022)](https://arxiv.org/abs/2201.12740), [PatchTST(2023)](https://arxiv.org/abs/2211.14730)가 있다.
  - 그리고 MLP 계열로는 [DLinear(2023)](https://arxiv.org/abs/2205.13504), [LightTS(2022)](https://arxiv.org/abs/2207.01186), [N-hits(2023)](https://arxiv.org/abs/2201.12886)가 있다.
- 지금까지는 그냥 decomposition하고 multiperiodicity 분석
  - decomposition : predictable component(seasonal, trend)와 complex temporal patterns를 분리
  - multiperiodicity analysis : mixed temporal variations을 주기가 다른 여러 components로 disentangle
- 본 논문의 intuition : sampling scales가 달라지면 temporal variations도 달라진다.
  - microscopic information을 담은 fine scale과 macroscopic information을 담고 있는 coarse scales에서의 변동이 joint하게 미래 변동을 결정한다.
  - multiscale mixing을 통해 scale에 따른 변동을 구분하고 그걸 다 고려해서 complementary한 예측을 할 수 있을 것
- **Past-Decomposable-Mixing** (PDM) : average downsampling을 통해 multiscale observations 생성
- **Future-Multipredictor-Mixing** (FMM) : multiscale에서의 seasonal, trend를 합쳐서 multiple predictors를 ensemble

## 2. Related Work

### 2.1. Temporal Modeling in Deep TSF

- Foundation backbones에 따라 4가지 계열로 구분 : CNN, RNN, Transformer, MLP
  - CNN, RNN-based deep models는 limited receptive field 때문에 long-term forecasting이 어려움
  - Transformer-based models는 long-term temporal depdendencies 파악
  - MLP-based models는 단순한 구조로도 복잡한 모델의 성능만큼 도달할 수 있음을 제시
- Temporal multiscale designs models이 있긴 하지만 [Pyraformer(2021)](https://openreview.net/forum?id=0EXmFzUn5I), [SCINet(2022)](https://arxiv.org/abs/2106.09305)
  - 예측 과정에서 multiscale information을 동시에 활용하지는 않음

### 2.2. Mixing Networks

- Computer Vision과 NLP 분야에서 사용되는 mixing [MLP-Mixer(2021)](https://arxiv.org/abs/2105.01601), [FNet(2022)](https://arxiv.org/abs/2105.03824)

## 3. TimeMixer

- P길이의 과거를 보고 F길이의 미래를 예측
- disentangled variations(past information extraction) & complementary forecasting(future prediction)

### 3.1. Multiscale Mixing Architecture

![그림1](/assets/img/timeseries/Timemixer/fig1.png)

- 먼저 complex variations를 disentangle하기 위해 downsampling한다.

  - $$\mathbf{x} \in \mathbb{R}^{P \times C}$$를 M개의 scale로 downsampling하면
  - $$\mathcal{X}=\left\{\mathbf{x}_0, \cdots, \mathbf{x}_M\right\} \text {, where } \mathbf{x}_m \in \mathbb{R}^{\left\lfloor \frac{P}{2^m}\right\rfloor \times C}, m \in\{0, \cdots, M\}$$를 얻는다.
  - $$C$$는 변수 개수
  - 즉 $$\mathbf{x}_0$$는 input 원본이고, $$\mathbf{x}_m$$은 m개씩 average pooling한 것이다.
  - 그리고 embedding layer 통과하면 $$\mathcal{X}^0=\operatorname{Embed}(\mathcal{X})$$를 얻는다.

- 이제 **Past-Decomposable-Mixing** (PDM, for mixing past information)

  $$x^l=\operatorname{PDM}\left(x^{l-1}\right),\quad l \in\{0, \cdots, L\}, \quad \mathbf{x}_m^l \in \mathbb{R}^{\left\lfloor\frac{P}{2^m}\right\rfloor \times d_{\text {model }}}$$

  - (자세한 건 다음 section 3.2.)

- 다음으로 **Future-Multipredictor-Mixing** (FMM, for future prediction)

  $$\widehat{\mathbf{x}}=\operatorname{FMM}\left(\mathcal{X}^L\right), \quad \widehat{\mathbf{x}} \in \mathbb{R}^{F \times C}$$

  - (자세한 건 다음 section 3.3.)

### 3.2. Past Decomposable Mixing

- Seasonal과 trend는 scale에 따라 다르게 나타나므로

  - Seasonal끼리 scale별로 구해서 mixing, trend끼리 scale별로 구해서 mixing

- $$l$$-번째 PDM block에서는
  - ts $$\mathcal{X}_l$$를 seasonal part $$\mathcal{S}^l=\left\{\mathbf{s}_0^l, \cdots, \mathbf{s}_M^l\right\}$$과 trend parts $$\mathcal{T}^l=\left\{\mathbf{t}_0^l, \cdots, \mathbf{t}_M^l\right\}$$로 decompose

  $$\begin{gathered}
  \mathbf{s}_m^l, \mathbf{t}_m^l=\text { SeriesDecomp }\left(\mathbf{x}_m^l\right), m \in\{0, \cdots, M\}, \\
  \mathcal{X}^l=\mathcal{X}^{l-1}+\text { FeedForward }\left(\operatorname{S-Mix}\left(\left\{\mathbf{s}_m^l\right\}_{m=0}^M\right)+\text { T-Mix }\left(\left\{\mathbf{t}_m^l\right\}_{m=0}^M\right)\right)
  \end{gathered}$$

  - $$\text { FeedForward(} \cdot)$$ contains two linear layers w/ GELU
  - $$\operatorname{S}-\operatorname{Mix}(\cdot), T-\operatorname{Mix}(\cdot)$$​는 지금부터 설명할 mixing

![그림2](/assets/img/timeseries/Timemixer/fig2.png)

- **Seasonal Mixing**
  - (Box & Jenkins, 1970)의 seasonality analysis에 따르면
    - larger periods는 smaller periods의 aggregation
    - 그러므로 residual하게 bottom-up approach 
    - (coarser scales의 seasonality를 위해 lower-level fine-scale 사용)
    - $$\text { for } m: 1 \rightarrow M \text { do: } \quad \mathbf{s}_m^l=\mathbf{s}_m^l+\text { Bottom-Up-Mixing }\left(\mathbf{s}_{m-1}^l\right)$$
      - $$\text { Bottom-Up-Mixing(} \cdot \text {) }$$: input dim $$\left\lfloor\frac{P}{2^{m-1}}\right\rfloor$$, output dim $$\left\lfloor\frac{P}{2^{m}}\right\rfloor$$인 two linear layers with GELU 

- **Trend Mixing**
  - seasonal parts와 반대로
    - detailed variations는 noise로 보일 뿐이고 coarser scales가 finer scales를 guide
    - 그러므로 residual하게 top-down mixing
    - $$\text { for } m:(M-1) \rightarrow 0 \text { do: } \quad \mathbf{t}_m^l=\mathbf{t}_m^l+\text { Top-Down-Mixing }\left(\mathbf{t}_{m+1}^l\right)$$
      -  $$\text { Top-Down-Mixing(} \cdot \text {) }$$ : : input dim $$\left\lfloor\frac{P}{2^{m+1}}\right\rfloor$$, output dim $$\left\lfloor\frac{P}{2^{m}}\right\rfloor$$인 two linear layers with GELU 

- 결론적으로 seasonality는 from fine to coarse, 반대로 trend는 coarse to fine하게 multiscale mixing in past information extraction

### 3.3. Future MultiPredictor Mixing

- $$L$$개의 PDM block으로 $$\mathcal{X}^L=\left\{\mathbf{x}_0^L, \cdots, \mathbf{x}_M^L\right\}, \mathbf{x}_m^L \in \mathbb{R}^{\left\lfloor\frac{P}{2^m}\right\rfloor \times d_{\text {model }}}$$를 얻었다.
- 서로 다른 scale인 $$\mathbf{x}_m^L$$들은 서로 다른 variations를 present하고 있기 때문에 모든 scale에서 예측을 하고 각각의 예측을 aggregate한다. (ensemble)
- $$\widehat{\mathbf{x}}_m=\operatorname{Predictor}_m\left(\mathbf{x}_m^L\right), m \in\{0, \cdots, M\}, \widehat{\mathbf{x}}=\sum_{m=0}^M \widehat{\mathbf{x}}_m$$
  - $$\widehat{\mathbf{x}}_m, \widehat{\mathbf{x}} \in \mathbb{R}^{F \times C}$$​
- $$\text { Predictor }_m(\cdot)$$은 one single linear layer인데 $$\left\lfloor\frac{P}{2^m}\right\rfloor$$길이의 과거 정보로부터 F 길이의 future를 regress
- FMM은 mixed multiscale series로 complementary forecasting하는  ensemble of multiple predictors

## 4. Experiments

- Summary

![그림11](/assets/img/timeseries/Timemixer/table1.png)

- Main results

![그림12](/assets/img/timeseries/Timemixer/table2.png)

- Ablations

![그림15](/assets/img/timeseries/Timemixer/table5.png)

- Decomposition과 multiscale의 각 components에 대한 visualization

![그림3](/assets/img/timeseries/Timemixer/fig3.png)

![그림4](/assets/img/timeseries/Timemixer/fig4.png)


## 5. Conclusion

- Empowered by Past-Decomposable-Mixing and Future- Multipredictor-Mixing blocks, TimeMixer took advantage of both disentangled variations and complementary forecasting capabilities.