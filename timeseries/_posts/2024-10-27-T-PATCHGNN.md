---
layout: post
related_posts:
  _
title: 
description: >
  [ICML 2024](https://openreview.net/pdf?id=UZlMXUGI6e)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# T-PATCHGNN: Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach (ICML 2024)

## Abstract

- Transformable Patching Graph Neural Networks (T-PATCHGNN)
  - transforms each univariate irregular time series into a series of transformable patches
  - local semantics capture와, inter-time series correlation modeling는 하면서
  - avoiding sequence **length explosion** in aligned IMTS (무슨 의미인지 1. introduction (3)에서 설명)

- Time-adaptive graph neural networks으로 time-varying adaptive graphs를 학습해서
  - dynamic intertime series correlation를 표현

## 1. Introduction

- Multivariate Time Series (IMTS)의 특징은 irregular sampling intervals and missing data
- Irregularity within the series and asynchrony 때문에 다루기 어려움
  - ODE로 풀려고 한 적은 있지만 numerical integration process으로 인해 computationally expensive
-  IMTS forecasting의 어려움에는 3가지 이유가 있음
  - 첫번째는 (1) irregularity in intra-time series dependency modeling
    - **varying time intervals** between adjacent observations이 the consistent flow of time series data를 방해
  - 두번째는 (2) asynchrony in intertime series correlation modeling
    - **misaligned at time** due to irregular sampling or missing data.
  - 가장 중요한 건 (3) sequence length explosion with the increase of variables
    - 아래 fig1처럼 "단 하나의 변수라도 기록된 time stamp"는 모두 존재하는 걸로 해버리면, 변수 개수가 늘어남에 다라 time stamps의 수가 너무 많아지는 문제. (이러한 방법을 canonical pre-alignment representation이라고 부름)

![그림1](/assets/img/timeseries/TPatchGNN/fig1.png)

- 그래서 본 논문에서 제시하는 T-PATCHGNN의 장점은
  - 첫째로 The independent patching process for each univariate irregular time series으로 representation에서 sequence length explosion의 risk를 없애고
  - 둘째로 local semantics를 잘 잡기 위해 putting each individual observation into patches with richer context
  - 셋째로 transformable patching 후에 IMTS is naturally aligned in a consistent patch-level temporal resolution

- 본 논문의 contribution은 :
  - New transformable patching method to transform each univariate irregular time series of IMTS into a series of variable-length yet time-aligned patches
  - transformable patching outcomes을 바탕으로,  time-adaptive graph neural networks를 제안
  - building a benchmark for IMTS forecasting evaluation

## 2, Related Works

### 2.1. Irregular Multivariate Time Series Forecasting

pass

### 2.2. Irregular Multivariate Time Series Representation

- 기존에는 time-aligned manner로 IMTS를 representation (pre-alignment representation method)
  - 즉 하나의 변수라고 기록된 time stamp는 존재하는 걸로 생각하니
  - sequence length that equals the number of all unique time stamps in IMTS
  - 예를 들어 변수 1은 1,3,5 시점에 기록되고 변수 2는 2,4,6 시점에 기록되면 unique time stamps의 개수는 6이 됨
  - sequence length explosion problem 발생

### 2.3. Graph Neural Networks for Multivariate Time Series

- 2018년 DCRNN, STGCN은 pre-defined graph structures를 사용해서 실제로 쓰기 어려웠고

- 2019년부터 data로부터 graph structures를 학습하는 방식을 사용
  - 하지만 IMTS에서는 잘 작동을 안 함. mimisalignment at times으로 인해 inter-time series correlation modeling이 잘 안 됨
- Raindrop(2021)[[paper review](https://lpppj.github.io/timeseries/2024-02-09-Raindrop)]
  - 이 문제를 propagation the asynchronous observations at all the timestamps로 해결하려고 했지만  sequence length explosion problem을 피할 수 없음

## 3. Preliminary

### 3.1. Problem Definition

### Definition 1

- Irregular Multivariate Time Series
  - $$\mathcal{O}=\left\{\mathbf{o}_{1: L_n}^n\right\}_{n=1}^N=\left\{\left[\left(t_i^n, x_i^n\right)\right]_{i=1}^{L_n}\right\}_{n=1}^N$$, where
  - $$N$$개의 변수가 있고 $$n$$번째 변수는 $$L_n$$개의 observations가 있고, $$n$$번째 변수의 $$i$$번째 변수의 값은 $$t_i^n$$

### Definition 2

- Forecasting Query $$q_j^n$$
  - $$j$$-th query on $$n$$-th variable to predict its corresponding value at a future time $$q_j^n$$

### Problem 1

- Irregular Multivariate Time Series Forecasting
  - IMTS $$\mathcal{O} =  \left\{\left[\left(t_i^n, x_i^n\right)\right]_{i=1}^{L_n}\right\}_{n=1}^N$$와 Forecasting query $$\mathcal{Q}=\left\{\left[q_j^n\right]_{j=1}^{Q_n}\right\}_{n=1}^N$$가 있을 때,
  - problem은 accurately forecast recorded values $$\hat{\mathcal{X}}=\left\{\left[\hat{x}_j^n\right]_{j=1}^{Q_n}\right\}_{n=1}^N$$ in correspondence to the forecasting queries
  - $$\mathcal{F}(\mathcal{O}, \mathcal{Q}) \longrightarrow \hat{\mathcal{X}}$$로 표현됨

### 3.2. Canonical Pre-Alignment Representation for IMTS

- 2.2. Irregular Multivariate Time Series Representation 참고

## 4. Methodology

![그림1](/assets/img/timeseries/TPatchGNN/fig2.png)

### 4.1. Irregular Time Series Patching

- 모든 univariate TS에 같은 patching operation을 하니까 변수 index 표기는 생략

### 4.1.1. TRANSFORMABLE PATCHING

- Time series patching이 forecasting에 좋은 방법이라는 건 알려진 사실. benefits in :
  - capturing local semantic information,
  - reducing computation and memory usage,
  - modeling longer-range historical observations
- 일반적으로 time series patching은 하나의 patch에 같은 숫자의 observations가 있는데,
  - IMTS에서 time intervals는 다양하기 때문에 이러한 방식이 적절하지 않음
- 그래서 patch에 같은 개수의 observataions가 아니라, unified time horizon이 들어가도록 함
  - patch 안에 들어가는 observations의 개수는 다를 수 있지만, ex) 2시간인 건 동일하도록
- patch는 $$\left[\mathbf{o}_{l_p: r_p}\right]_{p=1}^P$$로 표현되고 $$P$$

### 4.1.2. PATCH ENCODING

- **Continuous time embedding**
  - $$\phi(t)[d]=\left\{\begin{array}{lll}
    \omega_0 \cdot t+\alpha_0, & \text { if } & d=0 \\
    \sin \left(\omega_d \cdot t+\alpha_d\right), & \text { if } & 0<d<D_t
    \end{array}\right.$$.
    - where the $$\omega_d$$ and $$\alpha_d$$ are learnable parameters and $$D_t$$ is embedding's dimension
  - Concatenation하면 observations in the patch:
    - $$\mathbf{z}_{l_p: r_p}=\left[z_i\right]_{i=l_p}^{r_p}=\left[\phi\left(t_i\right) \| x_i\right]_{i=l_p}^{r_p}$$.
    - 이건 하나의 patch에 대한 표현이 되는 것 !

- **Transformable time-aware convolution**
  - input sequence의 길이에 맞게 (adaptively), generated parameters와 transformable filter size를 사용
  - $$\mathbf{f}_d=\left[\frac{\exp \left(\mathbf{F}_d\left(z_i\right)\right)}{\sum_{j=1}^{L_p} \exp \left(\mathbf{F}_d\left(z_j\right)\right)}\right]_{i=1}^{L_p}$$으로 표현됨
    - where $$L_p$$ is the sequence length of patch $$\mathbf{z}_{l_p: r_p}, \mathbf{f}_d \in \mathbb{R}^{L_p \times D_{i n}}$$ is the derived filter for $$d$$-th feature map, $$D_{i n}$$ is dimension of inputs, and $$\mathbf{F}_d$$ denotes the meta-filter that can be instantiated by learnable neural networks
    - 이건 filter의 parameters를 along the temporal dimension으로 normalizaing해서 consistent scaling 하겠다는 것
  - 위 식으로 $$D-1$$개의 filters를 사용해서 **latent patch embedding** $$h_p^c \in \mathbb{R}^{D-1}$$를 얻음 :
    - $$h_p^c=\left[\sum_{i=1}^{L_p} \mathbf{f}_d[i]^{\top} \mathbf{z}_{l_p: r_p}[i]\right]_{d=1}^{D-1}$$.
    - 이건  encoded transformable patches:
      - variable-length sequences에 따라 flexibility를 가지고
      - parameterization for varying time intervals을 하면서
      -  additional learnable filter parameters 없이 더 긴 시퀀스를 처리할 수 있음
    - 마지막으로 $$h_p=\left[h_p^c \| m_p\right]$$ 이렇게 patch에 masking을 덧붙여주는데,
      - $$m_p$$는 이 patch 안에 observations가 하나 이상 있다~를 indicator로 표현
    -  최종적으로 $$\mathbf{h}_{1: P}=\left[h_p\right]_{p=1}^P \in \mathbb{R}^{P \times D}$$를 얻는다.
      - 이건 $$P$$개의 patch를 $$D-1$$차원으로 표현하고 마지막에는 masking으로 indicator를 붙인 것

### 4.2. Intra- and Inter-Time Series Modeling

- 이제 이  transformable patching을 irregular time series를 intra- and inter-time series modeling하는지 알아보자

### 4.2.1. TRANSFORMER TO MODEL SEQUENTIAL PATCHES

- 위에서 구한 $$\mathbf{h}_{1: P}=\left[h_p\right]_{p=1}^P \in \mathbb{R}^{P \times D}$$를 Transformer에 넣는다.
- 먼저 positional encoding을 하고
  - $$\mathbf{x}_{1: P}^{t f, n}=\mathbf{h}_{1: P}^n+\mathbf{P E}_{1: P}$$.
- Q, K, V를 만들어서 MHA를 통과한다.
  - $$\mathbf{q}_h^n=\mathbf{x}_{1: P}^{t f, n} \mathbf{W}_h^Q$$ / $$\mathbf{k}_h^n=\mathbf{x}_{1: P}^{t f, n} \mathbf{W}_h^K$$ / $$\mathbf{v}_h^n=\mathbf{x}_{1: P}^{t f, n} \mathbf{W}_h^V$$ where $$\mathbf{W}_h^Q, \mathbf{W}_h^K, \mathbf{W}_h^V \in \mathbb{R}^{D \times(D / H)}$$
  - $$\mathbf{h}_{1: P}^{t f, n}=\|_{h=1}^H \operatorname{Softmax}\left(\frac{\mathbf{q}_h^n \mathbf{k}_h^{n T}}{\sqrt{D / H}}\right) \mathbf{v}_h^n \in \mathbb{R}^{P \times D}$$,

### 4.2.2. TIME-VARYING ADAPTIVE GRAPH STRUCTURE LEARNING

- 한 변수를 예측하기 위해서 다른 변수의 정보는 매우 유용할 수가 있음
- 하지만 IMTS에서는 misaligned at times으로 인해 correlation modeling이 어려움
  - 그렇다고 Raindrop처럼 하기엔 e sequence length explosion problem이 발생
- 그래서 **transformable patching**으로 해결
  - patch를 observations의 개수가 아니라 시간 길이를 기준으로 끊다보니
  - 각 변수는 같은 숫자의 patches로 이루어지니까
  - time-adaptive graph neural networks로 inter-time series correlation를 modeling할 수 있음
- 즉 IMTS의  dynamic correlations를 파악하기 위해서는
  - series of time-varying adaptive graphs를 학습하겠다는 것이고
  - 지금 문제는 variable embedding이 training에서는 update 가능하지만 inference에서는 static
  - 그러니 learnable $$\mathbf{E}_1^s, \mathbf{E}_2^s \in \mathbb{R}^{N \times D_g}$$를 사용해서
  - 우리가 지금까지 만들었던  time-varying patch embedding $$\mathbf{H}_p^{t f}=\left[\mathbf{h}_p^{t f, n}\right]_{n=1}^N \in \mathbb{R}^{N \times D}$$을
    -  static variable embedding으로 만들면 됨
- 그 gated adding operation은 다음과 같음
  - $$\begin{gathered}
    \mathbf{E}_{p, k}=\mathbf{E}_k^s+g_{p, k} * \mathbf{E}_{p, k}^d, \\
    \mathbf{E}_{p, k}^d=\mathbf{H}_p^{t f} \mathbf{W}_k^d, \\
    g_{p, k}=\operatorname{ReLU}\left(\tanh \left(\left[\mathbf{H}_p^{t f} \| \mathbf{E}_k^s\right] \mathbf{W}_k^g\right)\right) \\
    k=\{1,2\}
    \end{gathered}$$, where
    - $$\mathbf{W}_k^d \in \mathbb{R}^{D \times D_g}, \mathbf{W}_k^g \in \mathbb{R}^{\left(D+D_g\right) \times 1}$$ are learnable parameters
- 이제 time-varying adaptive graph structure를 다음과 같이 얻음 : $$\mathbf{A}_p=\operatorname{Softmax}\left(\operatorname{ReLU}\left(\mathbf{E}_{p, 1} \mathbf{E}_{p, 2}^T\right)\right)$$

### 4.2.3. GNNS TO MODEL INTER-TIME SERIES CORRELATION

- 다음으로 dynamic inter-time series correlation at a patch-level resolution을 얻음
  - $$\mathbf{H}_p=\operatorname{ReLU}\left(\sum_{m=0}^M\left(\mathbf{A}_p\right)^m \mathbf{H}_p^{t f} \mathbf{W}_m^{g n n}\right) \in \mathbb{R}^{N \times D}$$.
  - where $M$ is the number of layers for GNNs, and $\mathbf{W}_m^{g n n} \in$ $\mathbb{R}^{D \times D}$ are learnable parameters at $m$-th layer.

### 4.3. IMTS Forecasti

- 이제 final latent representation을 얻는다 :
  - $$\mathbf{H}=\text { Flatten }\left(\left[\mathbf{H}_p\right]_{p=1}^P\right) \mathbf{W}^f \in \mathbb{R}^{N \times D_o}$$, where  $$\mathbf{W}^f \in \mathbb{R}^{P D \times D_o}$$ are learnable parameters.
  - 각 변수마다 이 representation을 얻는다
-  n-번째 변수의 final latent representation $$\mathbf{H}^n \in \mathbf{H}$$과, forecasting query $$\left\{\left[q_j^n\right]_{j=1}^{Q_n}\right\}_{n=1}^N$$를 가지고 MLP에 넣는다
  - $$\hat{x}_j^n=\operatorname{MLP}\left(\left[\mathbf{H}^n \| \phi\left(q_j^n\right)\right]\right)$$.

- 모델은 각 변수의 예측의 MSE를 줄이는 방향으로 학습
  - $$\mathcal{L}=\frac{1}{N} \sum_{n=1}^N \frac{1}{Q_n} \sum_{j=1}^{Q_n}\left(\hat{x}_j^n-x_j^n\right)^2$$.

### 4.4. Analysis on Scalabil

- The average sequence length : $$L_{t p}=L_{a v g} \leq L_{\max } \leq L_{c p r} \leq N \times L_{a v g}$$, where
  - $$L_{\text {avg }}=\frac{1}{N} \sum_{n=1}^N L_n$$.

## 5. Experiments

### 5.1. Experimental Setup

- Dataset : 
  - PhysioNet, MIMIC, Human Activity, and USHCN
  - training, validation, and test sets adhering to ratios of 60%, 20%, and 20%
- Evaluation Metric : 
  - $$\begin{aligned}
    \text { MSE }&=\frac{1}{N} \sum_{n=1}^N \frac{1}{Q_n} \sum_{j=1}^{Q_n}\left(\hat{x}_j^n-x_j^n\right)^2, \\\text { MAE }&=
     \frac{1}{N} \sum_{n=1}^N \frac{1}{Q_n} \sum_{j=1}^{Q_n}\left|\hat{x}_j^n-x_j^n\right| .
    \end{aligned}$$.

#### 5.2. Main Results

![그림1](/assets/img/timeseries/TPatchGNN/table1.png)

### 5.3. Ablation Study

![그림1](/assets/img/timeseries/TPatchGNN/table2.png)

### 5.4. Scalability and Efficiency Analysis

![그림1](/assets/img/timeseries/TPatchGNN/table3.png)

### 5.5. Effect of Patch Size

![그림1](/assets/img/timeseries/TPatchGNN/fig4.png)

## 6. Conclusion

- Transformable Patching Graph Neural Networks (T-PATCHGNN)
  - achieved the alignment between asynchronous IMTS
    - by transforming each univariate irregular time series into a series of transformable patches with varying observation counts but maintaining unified time horizon resolution.
    - without a canonical pre-alignment representation process, preventing the aligned sequence length from explosively growing

