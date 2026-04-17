---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2024](https://arxiv.org/abs/2310.06625)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# iTransformer: Inverted Transformers are Effective for Time Series Forecasting (ICLR 2024)

## Abstract
- Transformer-based TS : larger lookback $$\to$$ performance degradation & computation explosion.
- iTransformer : attention과 feed-forward network(FFN)를 inverted dimension에서 적용
  ![사진1](/assets/img/timeseries/iTransformer/fig2'.jpeg){:.lead width="00" height="100"}
- 각 time point가 token이 되는 것이 아니라 각 series(variate)가 token이 된다.
- Inverted dimension만 다르고 이외의 Transformer의 components는 수정 없이 그대로 사용한다.

## 1. introduction
- Transformer가 다른 fields에서는 linear model보다 성능이 좋은데, multivariate ts forecasting에서는 not sutable하다. \
  (특히 time points 사이에 semantic 관계보다 numerical 관계가 강한 경우에는 그냥 simple linear layer의 성능이 더 좋았다.)
- 그 이유는 한 시점에 기록된 서로 다른 변수들의 값들이 하나의 token으로 기록되기 때문이다. 논문에서는 아래처럼 표현하고 있다. \
  (= embed multiple variates of the same timestamp into indistinguishable channels...) \
  (= the points of the same time step that represent different physical meanings recorded by inconsistent mesurements are embedded into one token ...)
- 이러한 single time step의 token 형태는 receptive field가 너무 좁아서 유용한 정보를 얻어내기가 어렵다. 
- 또한 시계열은 데이터의 순서가 중요한데, transformer는 permutation-invariant attention mechanism이라서 temporal dimension을 잘 잡지 못한다.

- 그래서 iTransformer는 각각의 variate를 독립적으로 token으로 embedding해서 receptive field를 늘려준다.(Patching의 extreme case)
- 그러면 token은 series의 global representation을 통합해서 variate-centric하다.
- FFN은 개별 변수에 대해 인코딩된 lookback series를 보고 예측할 수 있을 정도의 generalizable representation을 학습할 수 있다. 
- 본 논문의 contribution은 아래와 같다.
  - Transformer가 비효과적인 것이 아니라 아직 underexplored라서 잘못 사용되었으니 component를 개선한다.
  - 독립적인 개별 시계열을 token으로 간주하여 self-attention으로 multivariate correlations를 파악하고, FFN으로 forecasting을 위한 series-global representation을 학습한다.

## 2. Related Work
- TCN-based, RNN-base forecasters를 지나서 Transformer가 시퀀스 모델링과 확장 가능성으로 좋은 성능을 보여주며 많은 variant가 나왔다.
- Transformer의 variant는 component를 수정하는지, architecture를 수정하는지에 따라 4가지로 구분된다.
  ![사진2](/assets/img/timeseries/iTransformer/fig3.jpeg)
  - 1) Temporal dependency 모델링을 위한 attention module 수정 (Component adaptation)
  - 2) Linear model이 떠오르면서, transformer에서도 component나 architecture를 바꾸지 않고 Stationarization, Channel independence, Patching 등을 통해 효율적으로 성능 향상
  - 3) Transformer의 component와 architecture를 모두 수정하여 multivariate의 cross-time and cross-variate dependency를 파악
  - 4) iTransformer는 Transformer의 component는 그대로 가져오지만 inverted하게 가져온다. (architecture만 바뀜)
  
## 3. iTransformer
-  현실 시나리오에서는 각각의 variate마다 발생 시점과 기록 시점의 delay 정도가 다를 수 있기 때문에 시점 $$t$$에서 모든 variates가 관측되지 않을 수도 있다. 뿐만 아니라 각각의 variate마다 통계적 분포 자체가 다를 수도 있다.

### 3.1. Structure Overview
- ![사진3](/assets/img/timeseries/iTransformer/fig4.png)
- iTransformer는 Transformer의 encoder-only architecture (including the embedding, projection, and Transformer blocks)
- **Embedding the whole series as the token** : 한 시점에서 많은 변수들을 하나의 token으로 간주하면 attention map을 학습하기 어렵다는 것은 patching으로 respective field를 늘리는 방식들이 좋은 성능을 낸다는 것으로부터 알 수 있다. 그러므로 each time series가 token이 되어 해당 변수의 properties를 다루고, self-attention으로 mutual interactions를, feed-forward networks로 series representations를 처리한다.
- iTransformer가 예측하는 future series $$\hat{\mathbf{Y}}_{:, n}$$ based on $$\mathbf{X}_{:, n}$$의 formula :
  ![사진4](/assets/img/timeseries/iTransformer/formula1.png)
  - $$\mathbf{H}=\left\{\mathbf{h}_1, \cdots, \mathbf{h}_N\right\} \in \mathbb{R}^{N \times D}$$은 $$D$$차원의 embedded tokens $$N$$개이고 $$h$$의 아래첨자는 layer index이다.
  - Embedding $$\mathbb{R}^T \mapsto \mathbb{R}^D$$과 Projection $$\mathbb{R}^D \mapsto \mathbb{R}^S$$는 MLP가 한다.
  - Inverted dimension으로 sequence의 순서가 FFN에 저장되므로 position embedding이 더이상 필요하지 않다.
- **iTransformers** : Attention에 multivariate correlation 외에는 requirements가 없어서 variates가 많아질 때 효율적이다. 또한 training과 inference에서 token의 개수가 다를 수 있어서 variates의 개수에 대해 유연한 모델이다.

### 3.2. Inverted Transformer Components
- **Layer normalization** : Layer normalization은 훈련할 때 convergence speed and stability를 위한 것이다. 기존 Transformer에서는 한 시점에서 multivariate representation을 normalize했었는데, non-causal이나 앞서 언급한 delay를 고려하면 interaction noise의 원인이 될 수 있다. 그러므로 개별 variate를 normalize한다. 그러면 measurements(=variates, sensor, series)끼리의 불일치성도 해소된다.
  $$\text { LayerNorm }(\mathbf{H})=\left\{\left.\frac{\mathbf{h}_n-\operatorname{Mean}\left(\mathbf{h}_n\right)}{\sqrt{\operatorname{Var}\left(\mathbf{h}_n\right)}} \right\rvert\, n=1, \cdots, N\right\}$$
- **Feed-Forward network** : 각각의 variate token을 FFN에 태우면 universal approximation therem(Hornik, 1991)에 의해 시계열의 복잡한 representation을 추출할 수 있다. 이러한 inverted blocks를 쌓으면 observed를 encoding하고 future series를 decoding하는 과정을 [MLP](/timeseries/2024-02-16-DLinear)처럼 할 수 있다. (MLP 방식은 시계열의 amplitude, periodicity, frequency spectrums까지 학습할 수 있고, time point self-attention보다도 좋은 성능을 낼 수 있다.)
- **Self-attention** : Inverted dimenstion으로 self-attention을 계산하면 $$Q, K, V \in \mathbf R^{N \times d_k}$$를 linear projection으로 구한다. 사전에 feature-dimension으로 normalize를 해놓았으니 pre-Softmax score $$\mathbf{A}_{i, j}=\left(\mathbf{Q} \mathbf{K}^{\top} / \sqrt{d_k}\right)_{i, j} \propto \mathbf{q}_i^{\top} \mathbf{k}_j$$는 variate-wisecorrelation을 의미하고, whole score map $$\mathbf{A} \in \mathbb{R}^{N \times N}$$는 multivariate correlations btw paired variate tokens가 된다.

## 4. Experiments

### 4.1. Forecasting Results
- 7개의 데이터셋 사용(ECL, ETT, Exchange, Traffic, Weather, Solar-Energy, PEMS), 10개의 forecasting models과 비교
  - Transformer-based : Autoformer, FEDformer, Stationary, Crossformer, PatchTST
  - Linear-based : DLinear, TiDE, RLinear
  - TCN-based : SCINet, TimesNet
  ![사진5](/assets/img/timeseries/iTransformer/table1.png)
- SOTA였던 PatchTST는 변동이 심한 PEM 데이터를 처리하기 어렵고, 명시적으로 multivariate correlation을 파악하는 Crossformer보다 iTransformer의 성능이 뛰어나다.

### 4.2. iTransformer Generality
- **Performance promotion**
  ![사진6](/assets/img/timeseries/iTransformer/table2.png)
  - Transformer-based models에 inverted framework를 적용하여 성능을 비교한 결과 일관되게 성능이 향상되었다.
- **Variate generalization**
  ![사진7](/assets/img/timeseries/iTransformer/fig5.png)
  - 데이터의 20%만으로 training을 하더라도 100%로 training을 했을 때에 비해서 성능에 큰 차이가 없다는 것은 iTransformer가 효율적으로 훈련할 수 있는 모델임을 의미한다. 즉 unseen variates에 대해 generalization capability가 뛰어나다. 그 이유는 1) 각 variate를 token으로 embedding하니 variate의 개수에 제한이 없어지기 때문이고, 2) FFN이 각 token에 identically하게 적용되어 어떤 time series에서든 존재하는 본질적인 패턴을 학습할 수 있기 때문이다.
- **Increasing lookback length**
  ![사진8](/assets/img/timeseries/iTransformer/fig6.png)
  - Transformer-based models는 look-back window가 길어져도 성능 향상으로 이어지지 않았지만, inverted framework는 look-back window가 길어지면 성능이 향상된다.

### 4.3. Model Analysis
- **Ablation Study**
  ![사진9](/assets/img/timeseries/iTransformer/table3.png)
  - Time series의 multivariate correlation(Variate)와 series representation(Temporal)을 학습하기 위해 Attention을 FFN으로 바꿔도 보고 아예 안써보기도 하면서 iTransformer가 가장 좋은 성능을 낸다는 것을 확인하였다.
- **Analysis of multivariate Representations and Correlations**
  ![사진10](/assets/img/timeseries/iTransformer/fig7.png)
  - The centered kernel alignment (CKA)는 높을수록 similar representation을 의미하는데, inverted frameworks가 유의미하게 높다.
  - 또한 얕은 layer의 attention map은 input series와 상관관계가 높고, 깊은 layer의 attention map은 future series와 상관관계가 높다는 점에서 interpretable attention이다.
- **Efficient training strategy**
  ![사진11](/assets/img/timeseries/iTransformer/fig8.png)
  - 각 배치에서 변수의 일부만 사용하여 훈련하더라도 MSE가 안정적이고, sample ratio가 낮아짐에 따라 memory가 적게 사용되므로 효율적인 모델이라 할 수 있다.

## 5. Conclusion and Future work
- iTransformer는 각 series를 variate token으로 보고 multivariate correlation을 파악하기 위해 attention과 FFN을 사용하였다. 
- 실험 결과를 통해 기존 Transformer-based time series forecasters보다 뛰어날 뿐만 아니라 interpretable한 성능을 보여주었다.