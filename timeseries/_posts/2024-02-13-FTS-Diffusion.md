---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2024](https://openreview.net/pdf?id=CdjnzWsQax)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# FTS-Diffusion: Generative Learning for Financial Time Series with Irregular and Scale-invariant Patterns (ICLR 2024)

## Abstract
- Financial deep learning 모델을 훈련시키기 위한 데이터가 부족한데, 그렇다고 synthetic data를 만들어내려 하니 irregular and scale-invariant patterns 때문에 어려움이 있음
- 패턴이 irregular하다는 말은 패턴이 발생하는 간격이 일정하지 않아서 예측하기 어렵다는 것
- 패턴이 scale-invariant하다는 말은 scale을 변화시켜도 형태가 유지된다는 말인데, 특정 패턴이 다양한 너비(폭)나 높이(진폭)로 나타날 수 있다는 말이다. 또한 프랙탈 구조처럼 축소해도 비슷한 패턴이 보이게 된다.
- 본 논문에서는 irregular and scale-invariant patterns을 학습하고 생성하는 모델 FTS-Diffusion을 제시한다.

![사진1](/assets/img/timeseries/fts-diff/fig1.jpeg)
![사진2](/assets/img/timeseries/fts-diff/fig2.jpeg)

## 1. Introduction
- FTS-Diffusion은 3가지의 modules로 구성된다.
- Pattern recognition - Pattern generation - Pattern evolution
- 패턴을 인식하고, 패턴을 생성한 뒤, 패턴을 이어붙여서 하나의 time series를 만든다는 것이다. 기존 time series generation 모델들이 어려워하던 irregular and scale-invariant 패턴을 모델링할 수 있다.

## 2. Related Work
- 본 논문에서 제시하는 모델은 time series를 생성하는 모델이다. 생성 모델은 크게 VAE 계열, GAN 계열, Diffusion 계열이 있다.
  
![사진3](/assets/img/timeseries/fts-diff/gm.jpeg)

- 일반적으로 좋은 생성 모델을 3가지 특성으로 정의하는데, 세 가지 모두 뛰어난 모델은 없고 상대적인 장단점이 존재한다.
- Diffusion이 속도는 상대적으로 느리지만 높은 퀄리티와 다양성 측면에서 뛰어나 많은 주목 받았고, 본 논문에서도 diffusion을 사용한다.

## 3. Problem Statement
- 시계열 $$X={x_1,...,x_M}$$은 $$M$$개의 segments로 이루어지고 $$x_m=\left\{x_{m, 1}, \ldots, x_{m, t_m}\right\}$$ 각 segment의 길이는 $$t_m$$
- conditional distribution $$f(\cdot\mid p,\alpha,\beta)$$에서 샘플링을 하는 것이고, $$p$$는 패턴, $$\alpha$$는 duration, $$\beta$$는 magnitude이다.
- tuple $$(p,\alpha,\beta)$$는 하나의 state이고, 패턴끼리의 dynamic across를 모델링하기 위해 Markov chain을 사용한다. 즉 transition probability $$Q(p_j,\alpha_j,\beta_j \mid p_i,\alpha_i,\beta_i)$$를 통해 time series를 생성한다.
- 이제 FTS-Diffusion의 각 모듈을 다시 살펴보면 아래와 같다.
  - Pattern recognition : 패턴 $$p$$를 인식하고 반복되는 패턴의 구조 $$\cal P$$ 학습
  - Pattern Generation : conditional distribution $$f(\cdot\mid p,\alpha,\beta), \forall p \in \cal P$$ 학습
  - Pattern Evolution : pattern transition probability $$Q(p_j,\alpha_j,\beta_j \mid p_i,\alpha_i,\beta_i)$$ 학습

![사진4](/assets/img/timeseries/fts-diff/fig3.jpeg)

## 4. Framework

### (1) Pattern recognition
- 전체 time series를 여러 개의 segments로 나누고 비슷한 segments끼리 묶어 K개의 clusters를 만드는 알고리즘 (Scale-Invariant Subsequence Clustering, SISC)
- SISC는 각 segment의 length는, 가장 가까운 centriod와의 거리가 최소가 되는 segment length로 결정한다.
- 이 때 거리 metric은 일반적으로 사용하는 euclidean이 아니라 length나 magnitude에 구애받지 않는 dynamic time wraping (DTW)를 사용하였다.
- centroid initialization은 처음 1개만 랜덤하게 고른 뒤 먼 segment일수록 다음 centroid가 될 확률이 높도록 하였다. (k-Center-Greedy와 비슷)

![사진5](/assets/img/timeseries/fts-diff/fig4.jpeg)

### (2) Pattern generation
- 패턴에 gaussian noise를 씌우고 denoising gradient를 학습하는 DDPM의 방식을 사용하여 패턴을 생성하였다.
- Diffusion으로 생성된 패턴을 (scaling) autoencoder에 통과시켜 원하는 length로 transform한다.
- Objective를 아래 식으로 사용하여 diffusion 모델과 autoencoder를 같이 학습시킨다.
  $$\mathcal{L}(\theta)=\mathbb{E}_{x_m}\left[\left\|x_m-\hat{x_m}\right\|_2^2\right]+\mathbb{E}_{x_m^0, i, \epsilon}\left[\left\|\epsilon^i-\epsilon_\theta\left(x_m^{i+1}, i, p\right)\right\|_2^2\right]$$   

### (3) Pattern generation
- Pattern evolution network $$\phi$$는 현재 state가 주어졌을 때 다음 state에 올 패턴들의 확률을 학습한다.
  $$(\hat p_{m+1}, \hat \alpha_{m+1}, \hat \beta_{m+1}) = \phi(p_m, \alpha_m, \beta_m) $$
- Pattern evolution objective는 아래와 같다.
  $$\mathcal{L}(\phi)=\mathbb{E}_{x_m}\left[\ell_{C E}\left(p_{m+1}, \hat{p}_{m+1}\right)+\left\|\alpha_{m+1}-\hat{\alpha}_{m+1}\right\|_2^2+\left\|\beta_{m+1}-\hat{\beta}_{m+1}\right\|_2^2\right]$$

## 4. Experiments
- S&P500, GOOG, ZC=F(옥수수 선물) 데이터를 활용하였고, 자산 가격은 non-stationary random walk를 따른다고 알려져있으므로, 통계적 특성을 가지는 수익률(return)을 사용하였다.

![사진6](/assets/img/timeseries/fts-diff/table1.jpeg)

- 위 결과는 실제 return의 분포와 synthesized 분포의 적합도를 테스트하는 KS test와 AD test 결과이다.

![사진7](/assets/img/timeseries/fts-diff/fig6.jpeg)

- Mixed data(생성된 synthesized data + 실제 observed data)로 training을 하고 real data로 test를 했을 때에도(TMTR, TATR), mixed data의 비율이 달라졌을 때에도 예측 성능이 일정하다는 것으로부터 FTS-Diffusion으로 생성한 synthesized data가 observed data와 유사하다고 볼 수 있다.

## 5. Conclusion
- Pattern recognition : SISC designed to identify patterns
- Pattern generation : diffusion-based network to synthesize the segments of patterns
- Pattern evolution : assemble generated segments with proper temporal evoution

## Implementation
- under review