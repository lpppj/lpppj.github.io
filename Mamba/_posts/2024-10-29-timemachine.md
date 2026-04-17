---
layout: post
related_posts:
  _
title: 
description: >
  [ECAI 2024](https://arxiv.org/abs/2403.09898)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting (ECAI 2024)

## Abstract

- Long-term time-series forecasting(LTSF)에서 우리는 long-term dependencies를 capture하는데
  - linear scalability와 computational efficiency를 유지해야 함
- 본 논문에서 제시하는 TimeMachine은 Mamba를 활용하여
  - the unique properties of time series를 발견하여
    - multi-scales에서 salient contextual cues를 만들고
  - quadruple-Mamba architecture를 합쳐서
    -  channel-mixing and channel-independence를 한 번에 통합
  - 서로 다른 scales에서의 global / local contexts를 effective하게 selection할 수 있게 함

## 1. Introduction

- LTSF에서 Capturing long-term dependencies가 핵심
- **Linear model** : DLinear, TiDE
  - may not well capture long-range correlations
- **Transformer-based** : iTransformer, PatchTST, Crossformer
  - suffer from the quadratic complexity
- **state-space models (SSMs)**
  - inferring over very long sequences
  - context-aware selectivity
  - LTSF에서도 활용될 수 있는가
    - highly content- and context-selective SSM이 최근에 많이 나오고 있고
    - effectively representing the context in time series에 쓸 수 있을 것
- Transforemr-based approach에서는 each observation이나 sub-series, 아니면 time series를 token(patch)로 만드는데
  - SSM에서 이러한 접근을 그대로 쓰면 성능이 안나옴
  - **그래서 salient contextual cues tailored to SSM을 extract하는 것이 먼저 !**

- 기존에는 channel-mixing way도 있고 (ex. Informer, FEDformer, and Autoformer, ...)
  - channel-independence way도 있는데, (ex. PatchTST, TiDE, ...)
  - 본 논문에서는 unified architecture : applicable to both scenarios!
- 그리고 Time series에는 downsampling해도 temporal relations가 유지된다는 특징이 있으니
  - 모든 time points를 token으로 만드는 건 redundant하고, PatchTST처럼 patch를 사용하는 건 good
  - 하지만 pre-defined small patch는 fixed resolution에서의 context만 제공
  - 그러니 iTransformer처럼 whole look-back window를 token으로 만드는 것이 낫고
  - 하지만 iTransformer처럼 channel-independence에서는 select sub-token contents가 잘 안됨
  - 그러므로, SSM 쓰면 더 잘 될 것

- 그러니 본 논문에서는 TimeMachine을 제안
  - MTS를 2개의 scale에서 context-aware prediction하기 위해 SSM을 사용
    -  high, low resolution이라는 2개의 scale마다 2개의 mamba를 사용.
       - 하나는 global perspectives for the channel-mixing
       - 다른 하나는 both global and local perspectives for the channel-independence
  - 이렇게 4개의 SSM modules를 사용해서 channel-independent, -dependent를 통합하고
    - 즉 btw-channel correlation이 "있으면" 잡아내고 "없으면" independent 처럼
    - 다양한 scales에서 global and local contextual information을 효율적으로 selection

## 2. Related Works

- Non-Transformer-based Supervised Approaches
  - Classical methods : ARIMA, VARMAX, GARCH, RNN, ...
  - MLP-based models : DLinear, TiDE, RLinear, ...
  - CNN-based : TimesNet, Scinet, ...
- Transformer-based Supervised Learning methods
  - iTransformer, PatchTST, Crossformer, FEDformer, stationary, Flowformer, and Autoformer
  - time series를 token series로 만들고 self-attention
  - 하지만 quadratic time and memory complexity

## 3. Proposed Method

- input sequence $$\mathbf{x}=\left[x_1, \ldots, x_L\right]$$
  - $$x_t \in \mathcal{R}^M$$ representing a vector of $$M$$ channels at time point $$t$$
- **Normalization**
  - the original MTS $$\mathbf{x}$$ into $$\mathbf{x}^0=\left[\mathbf{x}_1^{(0)}, \cdots, \mathbf{x}_L^{(0)}\right] \in \mathcal{R}^{M \times L}$$, via $$\mathbf{x}^{(0)}=\operatorname{Normalize}(\mathbf{x})$$.
  - Here, Normalize $$(\cdot)$$ represents a normalization operation RevIN
- **Channel Mixing vs. Channel Independence**
  - PatchTST에서는 Channel Independence가 좋다고 하지만
    - 그건 length에 비해 channels가 많지 않을 때 이야기고,
    - channels가 많을 때에는 Channel Mixing이 더 낫다
  - TimeMachine은 "potentially" inter-channel correlation을 잡고
    - Channel Independence일 때에는 independence를 찾음
  - input의 shape은 BML, output은 BMT

- **Embedded Representations**
  - 2-stage embedded representation
  - $$\mathbf{x}^{(1)}=E_1\left(\mathbf{x}^{(0)}\right), \quad \mathbf{x}^{(2)}=E_2\left(D O\left(\mathbf{x}^{(1)}\right)\right)$$, where
    - $$E_1: \mathbb{R}^{M \times L} \rightarrow \mathbb{R}^{M \times n_1}$$ and $$E_2: \mathbb{R}^{M \times n_1} \rightarrow \mathbb{R}^{M \times n_2}$$은 MLP
    - DO는 dropout, (MLP 쓰니까 overfitting 방지)
  - 이렇게 input length에 상관없이 fixed-length tokens로 embedding

![그림1](/assets/img/Mamba/timemachine/fig1.png)

- **Integrated Quadruple Mambas** (fig1 보면서 이해하면 좋음)

  - $$E_1, E_2$$ 각각의 embedding level에서 2개의 mamba를 사용

    - $$E_1$$ level에서 사용되는 2개의 mamba의 input은 $$D O\left(\mathbf{x}^{(1)}\right)$$
    - $$E_2$$ level에서 사용되는 2개의 mamba의 input은 $$D O\left(\mathbf{x}^{(2)}\right)$$

  - 첫 번째 mamba block 안에서는 2개의 FC-layers가 linear projection하고

    - 하나만 1d causal conv와 SiLU activation 통과, 그리고 structured SSM으로 간다
    - 그 다음 남은 하나의 linear projection을 더하고 FC-layer를 한 번 더 태움
    - 이때 **continuous-time SSM**은 input sequence $$u(t)$$를 latente state $$h(t)$$를 통해 output $$v(t)$$로 보낸다.
      - 즉 $$d h(t) / d t=A h(t)+B u(t), \quad v(t)=C h(t)$$
        - $$h(t)$$ is $$N$$-dimensional ($$N$$은 state expansion factor)
        - $$u(t)$$ is $$D$$-dimensional ($$D$$는 dimension factor)
        - $$v(t)$$의 dimension도 $$D$$
        - $$A$$, $$B$$, $$C$$는 coefficient matrices of proper size
      - **여기서 $$A$$, $$B$$, $$C$$, 그리고 hidden state를 time interval $$\Delta$$에 대한 함수로 놓음**
      - 이것이 모델을 input에 adaptive하게 context selectivity를 강화하는 방법
        - 즉 $$h_k=\bar{A} h_{k-1}+\bar{B} u_k, \quad v_k=C h_k$$
        - where $$h_k, u_k$$, and $$v_k$$ are respectively samples of $$h(t), u(t)$$, and $$v(t)$$ at time $$k \Delta$$,
        - $$\bar{A}=\exp (\Delta A), \quad \bar{B}=(\Delta A)^{-1}(\exp (\Delta A)-I) \Delta B$$.

    - (continuous 말고) **SSM**은 $$B$$, $$C$$, $$\Delta$$가 input에 따라 달라짐
      - $$B, C \leftarrow \operatorname{Linear}_N(u)$$, $$\Delta \leftarrow \text{softplus}(parameter +Linear _D\left(\right. Linear \left.\left._1(u)\right)\right)$$
      - coefficient matrices는 current token을 보고 정보를 selectively propagate하게 함
      - **channel-mixing case**에서는 각 univariate가 token(dim=$$n_2$$)이 되고 
        - **Inner mambas**에서는 $$BMn_2$$이 나오는데
        - Left / right inner mamba의 k번째 변수의 output은 $$v_{L, k}, v_{R, k} \in \mathcal{R}^{n_2}$$
          - 둘을 더하고 embedding된 $$\mathbf{x}^{(2)}$$을 skip connection하면 $$\mathbf{x}^{(3)}=\mathbf{v}_L \bigoplus \mathbf{v}_R \bigoplus \mathbf{x}^{(2)}$$ (Element-wise addition)
          - 그 다음 linear mapping $$P_1: \mathbf{x}^{(3)} \rightarrow\mathbf{x}^{(4)} \in \mathcal{R}^{M \times n_1}$$
        - **Outer mambas**에서도 비슷하게
          - $$v_{L, k}^*, v_{R, k}^* \in \mathcal{R}^{n_1}$$구하고 $$\mathbf{x}^{(5)} \in \mathcal{R}^{M \times n_1}$$랑 해서 셋이 더함
      - **channel-independence**에서는 처음에 $$B M L \mapsto(B \times M) 1 L$$ 이렇게 reshape을 해서 마치 match가 $$BM$$개이고 univariates인 것처럼 처리
        - Outer든 inner이든
          - mamba 하나는 input dim =1, token length =$$n_1$$ or $$n_2$$,
          - 다른 하나는 input dim =$$n_1$$ or $$n_2$$, token length =1
        - 이렇게 하면 global context and local context 동시에 학습 가능하고
        - fine and coarse scales with high- and low-resolution 각각의 context 추출

  - Channel mixing은 변수 개수가 많을 때 하고 independence랑 switch하려면

    -  input sequence를 그냥 transposed하면 됨

- Output Projection

  - MLP 쓰고 $$P_1$$ performs a mapping $$\mathcal{R}^{M \times n_2} \rightarrow \mathcal{R}^{M \times n_1}$$, $$P_2$$는 $$\mathbb{R}^{M \times 2 n_1} \rightarrow \mathbb{R}^{M \times T}$$
  - Residual connection도 fig1처럼 해주고
  - Outer Mambas에서 나온 $$\mathbf{x}^{(5)}$$랑 Inner Mambas에서 나온 $$\mathbf{x}^{(4)}$$를 concat해서 사용하게 됨
    - 즉 $$\mathbf{x}^{(6)}=\mathbf{x}^{(5)} \|\left(\mathbf{x}^{(4)} \bigoplus \mathbf{x}^{(1)}\right)$$

## 4. Result Analysis

### 4.1. Datasets

- seven benchmark datasets extensively used for LTSF:
  - Weather, Traffic, Electricity, and four ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2)

### 4.2. Experimental Environment

- 본 논문에서 제시하는 TimeMachine을 11 SOTA models와 비교 :
  - including iTransformer, PatchTST, DLinear, RLinear, Autoformer, Crossformer, TiDE, Scinet, TimesNet, FEDformer, and Stationary

### 4.3. Quantitative Results

![그림1](/assets/img/Mamba/timemachine/fig2.png)

![그림1](/assets/img/Mamba/timemachine/fig3.png)

### 4.4. Qualitative Result

![그림1](/assets/img/Mamba/timemachine/table2.png)

## 5.  Hyperparameter Sensitivity Analysis and Ablation Study

### 5.1. Effect of MLPs’ Parameters (n1, n2)

- MLP의 size인 $$n_1, n_2$$를 다양하게 해봤는데 별 차이 없음 (fig5)
  - MLP에 heavily dependent하지 않다는 것

![그림1](/assets/img/Mamba/timemachine/fig5.png)

### 5.2. Sensitivity of Dropouts

- Dropout ratio 적당하게 0.7 사용

### 5.3. Ablation of Residual Connections

- Residual connections 쓰는 것이 좋더라

### 5.4. Effects of Mambas’ Local Convolutional Width

- 각 Mamba 안에도 parameters가 있을거니까 local convolutional kernel widths를 2 and 4로실험 해봤더니 2가 낫더라

### 5.5. Ablation on State Expansion Factor of Mambas

![그림1](/assets/img/Mamba/timemachine/fig6.png)

-  State Expansion Factor를 8부터 256까지 해봤는데 256이 제일 좋아서 defualt로 설정

### 5.6. Ablation on Mamba Dimension Expansion Factor

- dimension expansion factor ($$E$$)도 있었는데, 크게하면 메모리는 많이 먹는데 성능 향상으로 이어지지는 않아서 그냥 1로 둔다

## 6. Strengths and Limitations

- memory efficiency and stable performance across varying look-back and prediction lengths !
- Weather에서 1등 못한게 limitation. (....?ㅋㅋ)

## 7. Conclusion

- LTSF with linear scalability and small memory footprints !
- integrated quadruple-Mamba architecture
  - to predict with rich global and local contextual cues at multiple scales
  - $$\to$$ unifies channel-mixing and channel-independence situations