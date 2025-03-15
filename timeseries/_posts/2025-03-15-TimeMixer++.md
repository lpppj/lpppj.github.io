---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2025 Oral](https://openreview.net/pdf?id=1CLzLXSFNn)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis (ICLR 2025)

## Abstract

- TIMEMIXER++: general-purpose time series pattern machine (TSPM)
  - (1) multi-resolution time imaging(MRTI)
    - MRTI transforms multi-scale time series into multi-resolution time images, capturing patterns across both temporal and frequency domains.
  - (2) time image decomposition (TID)
    - TID leverages dual-axis attention to extract seasonal and trend patterns,
  - (3) multi-scale mixing (MCM)
    - MCM hierarchically aggregates these patterns across scales.
  - (4) multi-resolution mixing (MRM)
    - MRM adaptively integrates all representationsacross resolutions

## 1. Introduction

- The core of TSPMs is their ability **to recognize and generalize** time series patterns inherent in time series data
  - RNN: limitations like Markovian assumptions and inefficiencies...
  - TCN: fixed receptive field...
- **Unlike language** tasks where tokens usually belong to distinct contexts,
  - time series data often involve overlapping contexts at a single time point
  - ex. daily, weekly, and seasonal patterns occurring **simultaneously** !
- What capabilities must a model possess, and what challenges must it overcome, to function as a TSPM?
  - **Lower CKA similarity** indicates more **diverse** representations across layers
    - imputation and anomaly detection에 적합
  - **Higher CKA similarity**: consistent representations across layers
    - forecastingand classification tasks에 적합
  - $$\to$$ universal model flexible enough to adapt to **multi-scale and multi-periodicity** patterns across various analytical tasks, which may favor either diverse or consistent representations.
- TimeMixer++: to simultaneously capture intricate time series patterns across multiple scales in the **time** domain and various resolutions in the **frequency** domain

![그림1](/assets/img/timeseries/TimeMixer++/fig1.png)

## 2. Related Work

- Hierarchical Time Series Modeling
  - 기존에는 moving averages to discern seasonal and trend components
  - TimeMixer++는 disentangles seasonality and trend directly within the latent space via dual-axis attention
    - enhancing adaptability to a diverse range of timeseries patterns and task scenarios

## 3. TimeMixer++

![그림1](/assets/img/timeseries/TimeMixer++/fig2.png)

- (1) input projection, (2) a stack of Mixerblocks, and (3) output projection
- Multi-scale Time Series:
  - $$\mathbf{x}_0 \in \mathbb{R}^{T \times C}$$를 TimeMixer처럼 Conv로 downsampling해서 multi-scale representation
    - $$\mathbf{x}_m=\operatorname{Conv}\left(\mathbf{x}_{m-1}\right., \text{stride} \left.=2\right), \quad m \in\{1, \cdots, M\}$$.
    - $$X_{\text {init }}=\left\{\mathbf{x}_0, \cdots, \mathbf{x}_M\right\}$$, where $$\mathbf{x}_m \in \mathbb{R}^{\left\lfloor\frac{T}{2^m}\right\rfloor \times C}$$.

### 3.1. Structure Overview

- (1 ) **Input Projection**
  - 기존 TimeMixer에서는 channel-independence strategy
  - TimeMixer++에서는 Channel mixing $$\to$$ Embedding
    - **Channel mixing** to capture cross-variable interactions
      - **self-attention** to the variate dimensions at the **coarsest** scale $$\mathbf{x}_M \in \mathbb{R}^{\left\lfloor\frac{T}{2^M}\right\rfloor \times C}$$
        - 왜 coarsest냐면 가장 global한 context를 가지고 있을테니까
      - $$\to$$ integration of information across variables
        - $$\mathbf{x}_M=\text{Channel- Attn}\left(\mathbf{Q}_M, \mathbf{K}_M, \mathbf{V}_M\right)$$.
    - 그 다음에 **embed** all multi-scale time series into a deep pattern set
      - $$X^0=\left\{\mathrm{x}_0^0, \cdots, \mathrm{x}_M^0\right\}=\operatorname{Embed}\left(X_{\text {init }}\right)$$.

- (2) **MixerBlocks**
  - patterns across "scales in the time domain" and "resolutions in the frequency domain".
  - *MixerBlocks*에서는 **convert multi-scale time series into multi-resolution time images**
    - disentangle seasonal and trend patterns through time image decomposition
    - 그 다음aggregate these patterns across differentscales and resolutions
    - $$X^{l+1}=\operatorname{MixerBlock}\left(X^l\right)$$. 자세한건 3.2.

- (3) **Output Projection**
  - $$L \times \text{MixerBlocks}$$ 통해서 the multi-scale representation set $$X^{L}$$를 얻음
  - 각 scale의 정보를 처리하기 위한 multiple prediction heads 사용 (task-adaptive)
    - 그 다음 ensemble :$$\text { output }=\operatorname{Ensemble}\left(\left\{\operatorname{Head}_m\left(\mathbf{x}_m^L\right)\right\}_{m=0}^M\right)$$
      - $$\text{Emsemble}$$: Averaging or weighted sum
      - $$\text{Head}$$: Linear layer

### 3.2. MixerBlock

- 일단 residyal way.
  - $$(l+1)$$th block에서 the input is the multi-scale representation set $$\mathcal{X}^l$$
    - the forward propagation : $$x^{l+1}=\operatorname{LayerNorm}\left(X^l+\operatorname{MixerBlock}\left(X^l\right)\right)$$
- 이제 앞서 소개한
  - (1) multi-resolution time imaging (MRTI)
  - (2) time image decomposition (TID)
  - (3) multi-scale mixing (MCM)
  - and (4) multi-resolution mixing (MRM)
  - 에 대해서 자세히 알아보자

- **Multi-resolution Time Imaging** (MRTI)
  - step 1: 먼저 global interaction을 가지고 있는 the coarsest scale $$\mathbf{x}_M^l$$을 **FFT**
    - top-K frequencies with the highest amplitudes 얻음
    - $$\mathbf{A},\left\{f_1, \cdots, f_K\right\},\left\{p_1, \cdots, p_K\right\}=\operatorname{FFT}\left(\mathbf{x}_M^l\right)$$.
      - $$\mathbf{A}=\left\{A_{f_1}, \cdots, A_{f_K}\right\}$$ represents the unnormalized amplitudes
      - $$\left\{f_1, \cdots, f_K\right\}$$ are the top-K frequencies
      - $$p_k=\left[\frac{T}{f_k}\right], k \in\{1, \ldots, K\}$$ denotes the corresponding period lengths
  - step 2: 그 다음 padding 후 1D $$\to$$ 2D **Reshaping**
    - $$\begin{aligned}
      \operatorname{MRTI}\left(\mathcal{X}^l\right) & =\left\{\mathscr{L}_m^l\right\}_{m=0}^M=\left\{\mathbf{z}_m^{(l, k)} \mid m=0, \ldots, M ; k=1, \ldots, K\right\} \\
      & =\left\{\begin{array}{c}
      \left.\operatorname{Reshape}_{m, k}\left(\operatorname{Padding}_{m, k}\left(\mathbf{x}_m^l\right)\right) \mid m=0, \ldots, M ; k=1, \ldots, K\right\},
      \end{array}\right.
      \end{aligned}$$.
    - 길이 $$p_k \cdot\left\lceil\frac{\left\lfloor\frac{T}{2^m}\right\rfloor}{p_k}\right\rceil$$ 인 **time series** $$\to$$ 사이즈 $$p_k \times\left\lceil\frac{\left\lfloor\frac{T}{2^m}\right\rfloor}{p_k}\right\rceil$$인 **image** $$\mathbf{z}_m^{(l, k)}$$
      - $$f_{\mathrm{m}, \mathrm{k}}=\left\lceil\frac{\left\lfloor\frac{T}{2^m}\right\rfloor}{p_k}\right\rceil$$로 표기하겠음
- **Time Image Decomposition** (TID)
  - Time series patterns are inherently nested, with **overlapping scales and periods** $$\to$$ multi-resolution approach
  - 특정 scale and period에서의 image $$\mathbf{z}_m^{(l, k)} \in \mathbb{R}^{p_k \times f_{\mathrm{m} . \mathrm{k}} \times d_{\mathrm{model}}}$$
    - **Columns** in each image correspond to time series segments **within a period**
    - **Rows** represent consistent time points **across periods**
  - Dual-axis attention !
    - **Column-axis attention** ($$\text{Attention}_\text{col}$$) captures **seasonality within periods,**
    - **Row-axis attention** ($$\text{Attention}_\text{row}$$)  extracts **trend across periods**
  - 즉 seasonal, trend image는 각각
    - $$\begin{aligned}\mathbf{s}_m^{(l, k)}&=\text {Attention}_{\text{col }}\left(\mathbf{Q}_{\mathrm{col}}, \mathbf{K}_{\mathrm{col}}, \mathbf{V}_{\mathrm{col}}\right),\\\mathbf{t}_m^{(l, k)}&=\operatorname{Attention}_{\text {row }}\left(\mathbf{Q}_{\text {row }}, \mathbf{K}_{\text {row }}, \mathbf{V}_{\text {row }}\right)\end{aligned}$$.
      - where $$\mathbf{s}_m^{(l, k)}, \mathbf{t}_m^{(l, k)} \in \mathbb{R}^{p_k \times f_{\mathrm{m}, \mathrm{k}} \times d_{\text {model }}}$$
- **Multi-scale Mixing** (MCM)
  - Each period $$p_k$$에서 seasonal, trend image $$M+1$$개씩 얻음
    - $$\left\{\mathbf{s}_m^{(l, k)}\right\}_{m=0}^M \text { and }\left\{\mathbf{t}_m^{(l, k)}\right\}_{m=0}^M$$.
  - **Seasonal에서는 Longer patterns는 compositions of shorter ones**이므로 !
    - 2D conv in residual manner로
    - mix the **seasonal** patterns from **fine**-scale to **coarse**-scale (bottem-up)
      - $$\text { for } m: 1 \rightarrow M \text { do: } \quad \mathbf{s}_m^{(l, k)}=\mathbf{s}_m^{(l, k)}+2 \mathrm{D}-\operatorname{Conv}\left(\mathbf{s}_{m-1}^{(l, k)}\right)$$.
  - 반대로 **Trend에서는 coarser scales naturally highlight the overall trend**이므로
    - 2D transposed conv in residual manner로
    - mix the **trend** patterns from **coarse**-scale to **fine**-scale (top-down)
      - $$\text { for } m: M-1 \rightarrow 0 \text { do: } \quad \mathbf{t}_m^{(l, k)}=\mathbf{t}_m^{(l, k)}+2 \mathrm{D}-\operatorname{TransConv}\left(\mathbf{t}_{m+1}^{(l, k)}\right)$$.
  - 이렇게 seasonal, trend 각각을 섞고 나면
    - seasonal, trend는 aggregated $$\to$$ reshape back to 1D
    - $$\mathbf{z}_m^{(l, k)}=\operatorname{Reshape}_{2 D \rightarrow 1}\left(\mathbf{s}_m^{(l, k)}+\mathbf{t}_m^{(l, k)}\right), \quad m \in\{0, \cdots, M\}$$.
- **Multi-resolution Mixing** (MRM)
  - 마지막으로 each scale에서 $$K$$개의 periods를 섞어줘야 함
    - amplitudes = importance of each period,
    - 그러므로 $$\begin{aligned}\left\{\hat{\mathbf{A}}_{f_k}\right\}_{k=1}^K&=\operatorname{Softmax}\left(\left\{\mathbf{A}_{f_k}\right\}_{k=1}^K\right), \\\mathbf{x}_m^l&=\sum_{k=1}^K \hat{\mathbf{A}}_{f_k} \circ \mathbf{z}_m^{(l, k)}, \quad m \in\{0, \cdots, M\}\end{aligned}$$

## 4. Experiments

- (1) long-termforecasting,
- (2) univariate and (3) multivariate short-term forecasting,
- (4) imputation, (5) classification (6) anomaly detection,
- (7) few-shot and (8) zero-shot forecasting

### 4.1. Main Results

### 4.1.1. Long-Term Forecasting

![그림1](/assets/img/timeseries/TimeMixer++/table1.png)

### 4.1.2. Univariate Short-Term Forecasting

![그림1](/assets/img/timeseries/TimeMixer++/table2.png)

### 4.1.3. Multivariate Short-Term Forecasting

![그림1](/assets/img/timeseries/TimeMixer++/table3.png)

### 4.1.4. Imputation

![그림1](/assets/img/timeseries/TimeMixer++/table4.png)

### 4.1.5. Few-shot Forecasting

![그림1](/assets/img/timeseries/TimeMixer++/table5.png)

### 4.1.6. Zero-shot Forecasting

![그림1](/assets/img/timeseries/TimeMixer++/table6.png)

### 4.1.7. Classification and Anomaly Detection 

![그림1](/assets/img/timeseries/TimeMixer++/fig3.png)

### 4.2. Model Analysis 

- Ablation Study

![그림1](/assets/img/timeseries/TimeMixer++/table7.png)

- Representation Analysis

![그림1](/assets/img/timeseries/TimeMixer++/fig4.png)

## 5. Conclusion

- TIMEMIXER++: novel framework designed as a universal time seriespattern machine for predictive analysis
  - **Multi-resolution imaging**: constructs time images at various resolutions, enabling enhanced representation of temporal dynamics
  - **Dual-axis attention**: allows for effective decomposition of these time images, disentangling seasonal and trend components within deep representations
  - **Multi-scale and Multi-resolution mixing techniques**, fuses and extracts information across different hierarchical levels