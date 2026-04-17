---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2023](https://openreview.net/forum?id=vSVLM2j9eie)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting (ICLR 2023)

## Abstract

- Transformer-based models
  - focus on modeling the temporal dependency (cross-time dependency)
  - yet often omit the dependency among different variables (cross- dimension dependency)
- Crossformer는
  - Dimension-Segment-Wise (DSW) : MTS $$\to$$ 2d vector array로 만들고
  - Two-Stage Attention (TSA) : 2개의 attention을 거치는데 각각 cross-time and cross-dimension dependency를 학습한다.
  - Hierarchical Encoder-Decoder (HED) : 그리고 서로 다른 scales의 정보를 사용해서 coarse, fine한 정보 모두 활용하여 forecasting한다.



## 1. Introduction

- MTS에서는 cross-time dependency 뿐만 아니라 cross-dimension dependency도 중요한데, Transformer에서 cross-dim dependency를 반영하는 방법은 embedding 뿐이다.
- 본 논문에서는 cross-dim dependency를 explicitly하게 사용한다.
- Dimension-Segment-Wise (DSW) : the series(e.g. UTS)는 segments로 나뉘고, 각 segment는 feature vector가 된다. (series는 2d vector array가 된다.)
- Two-Stage Attention (TSA): 2d vector array로부터 cross-time and cross-dimension dependency 학습
- Hierarchical Encoder-Decoder (HED) : 각 layer에서 서로 다른 scale에 대한 dependency를 학습하게 된다.

## 2. Related Works

- **Multivariate Time Series Forecasting**
  - Statistical models : Vector auto-regressive(VAR), Vector auto-regressive moving average (VARMA)
  - Neural models : TCN, DeepAR, LSTnet(CNN+RNN), MTGNN, ...
- **Transformer-based model** : LogTrans, Informer, Autoformer, Pyraformer, FEDformer, Preformer, ...
- **Vision Transformers** : Transformer를 vision에서 사용할 때 썼던 patching 방식

## 3. Methodology

- $$\mathbf{x}_{1: T} \in \mathbb{R}^{T \times D}$$를 보고 $$\mathbf{x}_{T+1: T+\tau} \in \mathbb{R}^{\tau \times D}$$​ 예측하는 문제
  - $$\tau, T$$ is the number of time steps in the future and past, respectively
  - $$D>1$$ is the number of dimensions

### 3.1. Dimension-Segment-wise Embedding
![사진1](/assets/img/timeseries/crossformer/fig1.png)
- t시점의 모든 dimension의 data point $$\mathbf{x}_t \in \mathbb{R}^D$$를 $$\mathbf{h}_t \in \mathbb{R}^{d_{\text {model }}}$$로 embedding한다.
- $$\begin{aligned}
  \mathbf{x}_{1: T} & =\left\{\mathbf{x}_{i, d}^{(s)} \left\lvert\, 1 \leq i \leq \frac{T}{L_{\text {seg }}}\right., 1 \leq d \leq D\right\} \\
  \mathbf{x}_{i, d}^{(s)} & =\left\{x_{t, d} \mid(i-1) \times L_{\text {seg }}<t \leq i \times L_{\text {seg }}\right\} \\ \mathbf{h}_{i, d}&=\mathbf{E} \mathbf{x}_{i, d}^{(s)}+\mathbf{E}_{i, d}^{(p o s)} \end{aligned}$$
  - $$\mathbf{x}_{i, d}^{(s)} \in \mathbb{R}^{L_{\text {seg }}}$$  is the $$i$$-th segment in dimension $$d$$ with length $$L_{\text {seg }}$$
  - $$\mathbf{E} \in \mathbb{R}^{d_{\text {model }} \times L_{\text {seg }}}$$ : the learnable projection matrix
  - $$\mathbf{E}_{i, d}^{(\text {pos })} \in \mathbb{R}^{d_{\text {model }}}$$ : the learnable position embedding for position $$(i, d)$$. 
- $$\mathbf{H}=\left\{\mathbf{h}_{i, d} \mid, 1 \leq i \leq \frac{T}{L_{\text {seg }}}, 1 \leq d \leq D\right\}$$
  - where each $$\mathbf{h}_{i, d}$$ represents a univariate time series segment.
- 수식으로 표현하다보니 어려운데 아래 그림과 같고, $$\mathbf{H}$$는 오른쪽처럼 생겼다.
  ![사진2](/assets/img/timeseries/crossformer/myfig1.jpeg)

### 3.2. Two-Stage Attention Layer
- 이미지가 아니라 시계열이다보니 height와 width가 서로 바뀌면 의미가 달라지기 때문에 flatten시키면 안되고 바로 $$\mathbf{H}$$에 self-attention을 적용한다.
- **Cross-Time Stage**
  - $$\mathbf{Z}_{i, \text { : }}$$ : the vectors of all dimensions at time step $$i$$
  - $$\mathbf{Z}_{:, d}$$ the vectors of all time steps in dimension $$d$$
![사진3](/assets/img/timeseries/crossformer/myfig2.jpeg)
  - $$\begin{aligned} \hat{\mathbf{Z}}_{:, d}^{\text {time }}=\text { LayerNorm }\left(\mathbf{Z}_{:, d}+\operatorname{MSA}^{\text {time }}\left(\mathbf{Z}_{:, d}, \mathbf{Z}_{:, d}, \mathbf{Z}_{:, d}\right)\right) \\ \mathbf{Z}^{\text {time }}=\text { LayerNorm }\left(\hat{\mathbf{Z}}^{\text {time }}+\operatorname{MLP}\left(\hat{\mathbf{Z}}^{\text {time }}\right)\right) \end{aligned}$$
  - $$\mathbf{Z}^{time}$$이 다음 stage인 Cross-Dimension Stage의 input이 된다.
- **Cross-Dimension Stage**
  - $$\begin{aligned} \mathbf{B}_{i,:} & =\mathrm{MSA}_1^{\operatorname{dim}}\left(\mathbf{R}_{i,:}, \mathbf{Z}_{i,:}^{\text {time }}, \mathbf{Z}_{i,:}^{\text {time }}\right), 1 \leq i \leq L \\ \overline{\mathbf{Z}}_{i,:}^{\text {dim }} & =\mathrm{MSA}_2^{\text {dim }}\left(\mathbf{Z}_{i,:}^{\text {time }}, \mathbf{B}_{i,:}, \mathbf{B}_{i,:}\right), 1 \leq i \leq L \\ \hat{\mathbf{Z}}^{\text {dim }} & =\text { LayerNorm }\left(\mathbf{Z}^{\text {time }}+\overline{\mathbf{Z}}^{\text {dim }}\right) \\ \mathbf{Z}^{\text {dim }} & =\text { LayerNorm }\left(\hat{\mathbf{Z}}^{\text {dim }}+\operatorname{MLP}\left(\hat{\mathbf{Z}}^{\text {dim }}\right)\right) \end{aligned}$$
![사진4](/assets/img/timeseries/crossformer/fig2.png)
  - $$D$$가 클 때에는 router mechanism을 사용하여 fixed number $$c << D$$ vectors에 정보를 모았다가 다시 뿌려준다.
  - $$\mathbf{R} \in \mathbb{R}^{L \times c \times d_{\text {model }}}$$ : the learnable vector array serving as routers
  - $$\mathbf{B} \in \mathbb{R}^{L \times c \times d_{\text {model }}}$$ : the aggregated messages from all dimensions
  - $$\overline{\mathbf{Z}}^{\text {dim }}$$ : output of the router mechanism.
  - All time steps $$(1 \leq i \leq L)$$ share the same $$\mathbf{M S A}_1^{\text {dim }}, \mathbf{M S A}_2^{\text {dim }}$$
  - $$\hat{\mathbf{Z}}^{\text {dim }}, \mathbf{Z}^{\text {dim }}$$ : output of skip connection and MLP respectively

### 3.3. Hierarchical Encoder-Decoder
![사진5](/assets/img/timeseries/crossformer/fig3.png)
- Upper layer일수록 coarser scale을 사용한 정보를 얻고, 서로 다른 scale로 얻은 정보들로 예측한 값들은 final result에서 더해진다.
- **Encoder** : upper layer일수록 coarser scale을 사용한다는 말은 인접한 두 vector(segment)를 merge한다는 것과 같다.
- $$\mathbf{Z}^{e n c, l}=\operatorname{Encoder}\left(\mathbf{Z}^{e n c, l-1}\right)$$의 연산은 아래와 같다.
$$\begin{aligned} & \begin{cases}l=1: & \hat{\mathbf{Z}}^{e n c, l}=\mathbf{H} \\ l>1: & \hat{\mathbf{Z}}_{i, d}^{e n c, l}=\mathbf{M}\left[\mathbf{Z}_{2 i-1, d}^{e n c, l-1} \cdot \mathbf{Z}_{2 i, d}^{e n c, l-1}\right], 1 \leq i \leq \frac{L_{l-1}}{2}, 1 \leq d \leq D\end{cases} \\& \mathbf{Z}^{\text {enc,l}}=\operatorname{TSA}\left(\hat{\mathbf{Z}}^{\text {enc,l}}\right) \end{aligned}$$
  - $$\mathbf{H}$$ denotes the 2D array obtained by DSW embedding
  
  - $$\mathbf{Z}^{e n c, l}$$ denotes the output of the $$l$$-th encoder layer

  - $$\mathbf{M} \in \mathbb{R}^{d_{\text {model }} \times 2 d_{\text {model }}}$$ denotes a learnable matrix for segment merging
  - $$[\cdot]$$ denotes the concatenation operation
  - $$L_{l-1}$$ denotes the number of segments in each dimension in layer $$l-1$$
  - $$\hat{\mathbf{Z}}^{e n c, l}$$ denotes the array after segment merging in the $$i$$-th layer
  - $$\mathbf{Z}^{\text {enc }, 0}, \mathbf{Z}^{\text {enc }, 1}, \ldots, \mathbf{Z}^{\text {enc }, N},\left(\mathbf{Z}^{\text {enc }, 0}=\mathbf{H}\right)$$ is used to represent the $$N+1$$ outputs of the encoder
- **Decoder** : Emcoder에서 얻은 $$N+1$$개의 feature array가 있으면, $$N+1$$개의 layers로 예측한다.
  - decoder의 process : $$\mathbf{Z}^{\text {dec, } l}=\operatorname{Decoder}\left(\mathbf{Z}^{\text {dec, },-1}, \mathbf{Z}^{\text {enc, },}\right)$$
$$\begin{aligned} & \left\{\begin{array}{lll} l=0: & \tilde{\mathbf{Z}}^{\text {dec }, l}=\operatorname{TSA}\left(\mathbf{E}^{(d e c)}\right) \\ l>0: & \tilde{\mathbf{Z}}^{\text {dec, }, l}=\operatorname{TSA}\left(\mathbf{Z}^{\text {dec, },-1}\right) \end{array}\right. \\ & \overline{\mathbf{Z}}_{:, d}^{\text {dec, }, l}=\operatorname{MSA}\left(\tilde{\mathbf{Z}}_{:, d}^{\text {dec, }, l}, \mathbf{Z}_{:, d}^{e n c, l}, \mathbf{Z}_{:, d}^{e n c, l}\right), 1 \leq d \leq D \\ & \hat{\mathbf{Z}}^{\text {dec, } l}=\text { LayerNorm }\left(\tilde{\mathbf{Z}}^{\text {dec, }, l}+\overline{\mathbf{Z}}^{\text {dec, } l}\right) \\ & \mathbf{Z}^{\text {dec,l}}=\text { LayerNorm }\left(\hat{\mathbf{Z}}^{\text {dec, }, l} \operatorname{MLP}\left(\hat{\mathbf{Z}}^{\text {dec,l}}\right)\right) \\ & \end{aligned}$$
  - $$\mathbf{E}^{(\text {dec })} \in \mathbb{R}^{\frac{\tau}{L_{s e g}} \times D \times d_{\text {model }}}$$ denotes the learnable position embedding for decoder
  - $$\tilde{\mathbf{Z}}^{\text {dec,l } l}$$ is the output of TSA
  - The MSA layer takes $$\tilde{\mathbf{Z}}_{:, d}^{d e c,l }$$ as query and $$\mathbf{Z}_{:, d}^{e n c, l}$$ as the key and value to build the connection between encoder and decoder
  - The output of MSA is denoted as $$\overline{\mathbf{Z}}_{:, d}^{\text {dec, },} . \hat{\mathbf{Z}}^{\text {dec,l}, ~} \mathbf{Z}^{\text {dec, }, l}$$ denote the output of skip connection and MLP respectively.
  - $$\mathbf{Z}^{\text {dec, 0},}, \mathbf{Z}^{e n c, 1}, \ldots, \mathbf{Z}^{\text {dec, } N}$$ : is used to represent decoder output
  - **Linear projection** : 각 layer에서는 linear projection으로 prediction을 만들고, 각 layer의 prediction을 다 더하면 최종 prediction이 된다.
$$\begin{gathered} \text { for } l=0, \ldots, N: \mathbf{x}_{i, d}^{(s), l}=\mathbf{W}^l \mathbf{Z}_{i, d}^{\text {dec,l }} \quad \mathbf{x}_{T+1: T+\tau}^{\text {pred, } l}=\left\{\mathbf{x}_{i, d}^{(s), l} \left\lvert\, 1 \leq i \leq \frac{\tau}{L_{\text {seg }}}\right., 1 \leq d \leq D\right\} \\ \mathbf{x}_{T+1: T+\tau}^{\text {pred }}=\sum_{l=0}^N \mathbf{x}_{T+1: T+\tau}^{\text {pred, }}\end{gathered}$$
    - $$\mathbf{W}^l \in \mathbb{R}^{L_{\text {seg }} \times d_{\text {model }}}$$ : learnable matrix to project a vector to a ts segment

## 4. Experiments
### 4.1. Protocols
- Dataset : 1) ETTh1 (Electricity Transformer Temperature-hourly), 2) ETTm1 (Electricity Transformer Temperature-minutely), 3) WTH (Weather), 4) ECL (Electricity Consuming Load), 5) ILI (Influenza-Like Illness), 6) Traffic
- Baselines : 1) LSTMa (Bah- danau et al., 2015), 2) LSTnet (Lai et al., 2018), 3) MTGNN (Wu et al., 2020), and recent Transformer-based models for MTS forecasting: 4) Transformer (Vaswani et al., 2017), 5) In- former (Zhou et al., 2021), 6) Autoformer (Wu et al., 2021a), 7) Pyraformer (Liu et al., 2021a) and 8) FEDformer (Zhou et al., 2022)
### 4.2. Main Results
![사진6](/assets/img/timeseries/crossformer/table1.png)
### 4.3. Ablation Study
![사진7](/assets/img/timeseries/crossformer/table2.png)
### 4.4. Effect of Hyper-parameters
![사진8](/assets/img/timeseries/crossformer/fig4.png)
### 4.5. Computational Efficiency Analysis
![사진9](/assets/img/timeseries/crossformer/table3.png)

### 5. Conclusion
- Crossformer : Transformer-based model utilizing cross-dimension dependency for MTS forecasting
- **Dimension-Segment-Wise (DSW)** embedding embeds the input data into a 2D vector array to preserve the information of both time and dimension
- **The Two-Stage-Attention (TSA)** layer is devised to capture the cross-time and cross- dimension dependency of the embedded array
- Using DSW embedding and TSA layer, a **Hierarchical Encoder-Decoder (HED)** is devised to utilize the information at different scales