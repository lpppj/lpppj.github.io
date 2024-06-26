---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2024](https://openreview.net/forum?id=vpJMJerXHU)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis (ICLR 2024)

## Abstract
- 최근 Transformer-based 모델과 MLP-based 모델이 time series에서 우위에 있지만, 본 논문에서는 convolution을 time series에서 사용하는 모델 ModernTCN을 제안한다.
- Time series의 5개 mainstream task (long-term and short-term forecasting, imputation, classification and anomaly detection)에서 SOTA
- Convolution의 sharing params로 효율적이면서도 넓은 receptive fields를 가진다.

## 1. Introduction
- Transformer-based 모델과 MLP-based 모델이 우위에 있는 이유는 global한 effective receptive fields (ERFs)가 cross-time dependency를 파악하기 때문이다.
- 하지만 지금까지 convolution in time series 연구들은 그저 모델의 구조를 복잡하게 해왔고, 이와 다르게 본 논문에서는 convolution 자체를 업데이트해서 ERF를 키운다.
- 왜냐하면 CV에서는 이미 Transformer을 보고 convolution을 optimizing하려고 시도하고 있기 때문이다.
- 시계열에서 convolution을 쓴다는 것은 cross-time and cross-variable dependency를 포착하겠다는 의미이다.

## 2. Related Works
- MICN(2023), SCINet(2023) 등 최근까지도 convolution을 time series에 활용하려는 시도가 많았지만 long-term dependency가 중요한 time series에서 limited ERFs는 transformer를 이길 수가 없었다.
- CNN(2017) 이후 ViTs(2020)이 등장했고, ViTs를 따라잡기 위해 다양한 modern convolution이 등장하는데, 예를 들면 conv block을 transformer와 비슷하게 하거나 (ConvNeXt, 2022), kernel size를 51 $$\times$$ 51로 늘려버리기도 했다. (SLaK, 2022)
- 본 논문에서는 conv를 time series에 쓰기 위해 1D conv를 수정한 ModernTCN을 제안한다.

## 3. ModernTCN

### 3.1. Modernize the 1D Convolution block
![사진1](/assets/img/timeseries/modernTCN/fig2.jpeg)

- 1D conv를 Figure-2:(b)처럼 DWConv(depth-wise)와 ConvFFN(feed-forward NN)으로 re-design하였다.
  - DWconv는 transformer의 self-attention와 같은 역할 : learning the temporal information among tokens on a **per-feature** basis
  - ConvFFN은 transformer의 FFN과 같은 역할 : learn the new feature representation of each token **independently**
- 위 디자인은 temporal and feature information을 분리한다. 이것이 jointly mix하던 tranditional conv와의 차이점이다.
- 하지만 multivariate time series에서는 cross-variable information도 중요하니 추가적인 수정이 필요하긴 하다.

### 3.2. Time series related Modifications
- CV에서는 RGB $$\to$$ D-dim embedding 하지만, 그대로 특정 t시점에서 M개 변수 $$\to$$ D-dim embedding하면 안된다.
  - RGB차이보다 t시점에서 M개 변수 사이의 차이가 더 크고, cross-variable dependency를 반영 못하기 때문
- 그래서 아래와 같은 방식으로 patchify embedding을 거친다.
  - 1) $$X_{in} \in \mathbb R^{M\times L}$$을 $$X_{in} \in \mathbb R^{M\times 1\times L}$$로 unsqueeze
  - 2) $$X_{in}$$ 뒤에 $$P-S$$만큼 패딩 ($$P$$는 patch size, $$S$$는 stride)
  - 3) 1D conv를 통과, 각 patch는 D차원으로 embedding
  - 그림으로 표현하면 아래와 같다.
    ![사진2](/assets/img/timeseries/modernTCN/myfig1.jpeg)
  - 예시로 이해해보자. patch size가 10이고 stride가 2이므로 총 50개의 patch를 보게 되므로 N=50이 된다. 
    ![사진3](/assets/img/timeseries/modernTCN/myfig2.png)
- DWConv는 feature와 variable 모두에 대해 independent하게, 그리고 kernel을 크게 해서 ERFs를 넓게 가져가 temporal information을 포착하도록 했다.
- ConvFFN은 information across feature and variable을 섞는 역할을 해야 하는데, 연산의 효율을 위해 jointly하게 학습하기보다는 두 개의 ConvFFN으로 decople했다.
  - ConvFFN 1 : learning the new feature representations per variable
  - ConvFFN 2 : learning the cross-variable dependency per feature

### 3.3 Overall Structure
- $$\mathbf{Z}=\operatorname{Backbone}(\mathbf X_{emb})$$, Backbone($$\cdot$$)은 ModernTCN을 쌓아서 만든 구조이다.
- 즉 다음과 같이 표현할 수 있다.
- $$\mathbf{Z}_{i+1}=\operatorname{Block}\left(\mathbf{Z}_i\right)+\mathbf{Z}_i$$ , 즉 $$\mathbf{Z}_i= \begin{cases}\mathbf{X}_{e m b} & , i=1 \\ \operatorname{Block}\left(\mathbf{Z}_{i-1}\right)+\mathbf{Z}_{i-1} & , i>1\end{cases}$$, 이 때 Blcok($$\cdot$$)은 ModernTCN block이다.


## 4. Experiments
![사진4](/assets/img/timeseries/modernTCN/fig3.jpeg)
- Time series의 5가지 mainstream analysis task에서 performance, efficiency 측면에서 SOTA를 달성했다.

## 5. Model Analysis
- Performance, efficiency 측면에서 ModernTCN은 SOTA를 달성했다.
- MLP-based time series model TimesNet(2023)도 ModernTCN만큼 성능이 좋은데, 그 이유가 두 모델 모두 CV 분야에서 convolution을 사용하는 아이디어에서 영감을 얻었기 때문이다.
- 다만, TimesNet은 conv를 사용하기 위해 1D time series를 2D 공간으로 보낸거고, ModernTCN은 conv 자체를 1D time series에 사용할 수 있도록 modernize했기 때문에 training speed가 빠르다.
 
## 6. Conclusion and Future Work
- 본 논문의 contribution은 단순히 conv-based model로 transformer-based 모델보다 좋은 성능을 냈다 정도가 아니라, 다시 한 번 time series에 다양한 conv-based models가 등장할 수 있음을 의미한다는 것이다.