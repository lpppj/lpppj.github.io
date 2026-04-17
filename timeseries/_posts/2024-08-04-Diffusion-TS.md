---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2024](https://arxiv.org/pdf/2403.01742)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# Diffusion-TS: Interpretable Diffusion for General Time Series Generation (ICLR 2024)

## Abstract

- Diffusion-TS: uses an encoder-decoder transformer with disentangled temporal representations
- train the model to directly reconstruct the **sample** instead of the **noise** in each diffusion step

## 1. Introduction

- Synthesizing realistic time series data는 데이터 공유가 개인정보 침해로 이어질 수 있는 사례에서의 솔루션
- 지금까지 Diffusion을 활용한 time series generation은 대부분 task-agnostic generation
  - 첫번째 문제는 RNN-based Autoregressive 방식: limited long-range performance due to error accumulation and slow inference speed
  - 두번째 문제는 diffusion process에서 noise를 추가할 때 시계열의 combinations of independent components(trend, seasonal, ...)이 망가지는 문제 (특히 주기성이 뚜렷한 경우 interpretability가 부족 [Liu et al., (2022)](https://openreview.net/pdf?id=rdjeCNUS6TG))
- **본 논문에서는 Transformer를 활용하여 trend와 seasonal을 non-autoregressive하게 생성**
  -  by imposing different forms of constraints on different representations.
- For Reconstruct the **samples** rather than the **noises** in each diffusion step, Fourier-based loss 사용

## 2. Problem Statement

- N개로 이루어진 데이터셋 $$D A=\left\{X_{1: \tau}^i\right\}_{i=1}^N$$​
  - where $$X_{1: \tau}=\left(x_1, \ldots, x_\tau\right) \in \mathbb{R}^{\tau \times d}$$
- 목표는 Gaussian vectors $$Z_i=\left(z_1^i, \ldots, z_t^i\right) \in \mathbb{R}^{\tau \times d \times T}$$를 DA와 비슷한 $$\hat{X}_{1: \tau}^i=G\left(Z_i\right)$$로 바꾸는 Generator $$G$$를 학습하는 것
- Time series model은 trend와 여러 개의 seasonality로 구성 : $$x_j=\zeta_j+\sum_{i=1}^m s_{i, j}+e_j$$
  - where $$j=0,1, \ldots, \tau-1$$
  - $$x_j$$ : observed time series
  - $$\zeta_j$$: trend component
  - $$s_{i,j}$$: $$i$$-th seasonal component
  - $$e_j$$: remainder part (contatins the noise and some outliers at time t)

## 3. Diffusion-TS: Interpretable Diffusion for Time Series

- 이러한 interpretable decomposition architecture의 근거는 3가지
  - 첫째, disentangled patterns in the diffusion model은 아직 연구되지 않음
  - 둘째, specific designs of architecture and objective 덕분에 interpretable
  - 셋째, explainable disentangled representations 덕분에 complex dynamics 파악

### 3.1. Diffusion Framework

![그림1](/assets/img/timeseries/Diffusion-TS/fig1.png)

- Forward process
  - $$x_0 \sim q(x)$$에서 점점 noisy into Gaussian noise $$x_T \sim \mathcal{N}(0, \mathbf{I})$$
  - Parameterization: $$q\left(x_t \mid x_{t-1}\right)=\mathcal{N}\left(x_t ; \sqrt{ } 1-\beta_t x_{t-1}, \beta_t \mathbf{I}\right) \text { with } \beta_t \in(0,1)$$

- Reverse process

  - 반대로 $$p_\theta\left(x_{t-1} \mid x_t\right)=\mathcal{N}\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right)$$

  - MSE: $$\mathcal{L}\left(x_0\right)=\sum_{t=1}^T \underset{q\left(x_t \mid x_0\right)}{\mathbb{E}}\left\|\mu\left(x_t, x_0\right)-\mu_\theta\left(x_t, t\right)\right\|^2$$
    - where $$\mu\left(x_t, x_0\right) \text { is the mean of the posterior } q\left(x_{t-1} \mid x_0, x_t\right)$$

### 3.2. Decomposition Model Architecture

![그림2](/assets/img/timeseries/Diffusion-TS/fig2.png)

- Noisy sequence가 encoder 통과해서 decoder로 들어옴 (초록색)
- Decoder는 multilayer structure, 각 layer에는 **Transformer Block**, **FFN**, **Trend and Fourier synthetic layer**가 포함됨
- 각 layer는 시계열의 각 component를 생성하는 역할
  - component에 해당하는 inductive bias를 각 layer에 반영해줌으로써 학습이 쉬워짐
  - Trend representation captures the intrinsic trend which changes gradually and smoothly
  - Seasonality representation illustrates the periodic patterns of the signal
  - Error representation characterizes the remaining parts after removing trend and periodicity

- $$w_{(\cdot)}^{i, t}$$ where $$i \in 1, \ldots, D$$는 $$i$$번째 decoder block에서의 diffusion step $$t$$를 의미

### Trend Synthesis

- smooth underlying mean of the data, which aims to model slow-varying behavior
- 그러므로 Trend $$V_{t r}^t$$를 위해 Polynomial regressor 사용
  - $$V_{t r}^t=\sum_{i=1}^D\left(C \cdot \operatorname{Linear}\left(w_{t r}^{i, t}\right)+\mathcal{X}_{t r}^{i, t}\right)$$ where $$C=\left[1, c, \ldots, c^p\right]$$
  - $$\mathcal{X}_{t r}^{i, t}$$는 the mean value of the output of the $$i$$​-th decoder block
  - $$C$$는 slow-varying poly space인데, matrix of powers of vector $$c=[0,1,2, \ldots, \tau-2, \tau-1]^T / \tau$$
  - $$p$$는 small degree (e.g. $$p$$​=3) to model low frequency behavior

### Seasonality & Error Synthesis

- 이제 Trend, Seasonality, Error 모두 생각해보자.
- **결국 문제는 noisy input $$x_t$$에서 seasonal patterns를 구분해내는 것 !**
- 푸리에 시리즈의 trigonometric representation of seasonal components를 기반으로 Fourier bases를 활용한 Fourier synthetic layers에서 seasonal component 파악

![그림456](/assets/img/timeseries/Diffusion-TS/fomula456.png)

- $$A_{i, t}^{(k)}, \Phi_{i, t}^{(k)}$$ are the phase, amplitude of the $$k$$-th frequency after the DFT $$\mathcal F$$ repectively
- $$f_k$$는 Fourier frequency of the corresponding index $$k$$
- 결국 the Fourier synthetic layer는 진폭(amplitude)이 큰 frequency를 찾고, 그 frequency들만 IDFT.
  - 그걸 seasonality로 본다. (Pathformer랑 같은 방식)

- 최종적으로 original signal: $$\hat{x}_0\left(x_t, t, \theta\right)=V_{t r}^t+\sum_{i=1}^D S_{i, t}+R$$​
  - $$R$$: output of the last decoder block, which can be regarded as the sum of residual periodicity and other noise.

### 3.3 Fourier-based Traning Objective

- $$\hat{x}_0\left(x_t, t, \theta\right)$$를 directly estimate
  - Reverse process: $$x_{t-1}=\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \hat{x}_0\left(x_t, t, \theta\right)+\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} x_t+\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t z_t$$
  - where $$z_t \sim \mathcal{N}(0, \mathbf{I}), \alpha_t=1-\beta_t \text { and } \bar{\alpha}_t=\prod_{s=1}^t \alpha_s$$​
- Reweighting strategy: $$\mathcal{L}_{\text {simple }}=\mathbb{E}_{t, x_0}\left[w_t\left\|x_0-\hat{x}_0\left(x_t, t, \theta\right)\right\|^2\right], \quad w_t=\frac{\lambda \alpha_t\left(1-\bar{\alpha}_t\right)}{\beta_t^2}$$​
  - where $$\lambda$$ is constant (i.e. 0.01)
  - 즉 small t에서 down-weighted, 모델이 larger diffusion step에 집중하도록 만듬
- Fourier-based loss term이 time serie reconstruction에서는 더 좋다 [Fons et al. (2022)](https://arxiv.org/pdf/2208.05836)
  - : $$\mathcal{L}_\theta=\mathbb{E}_{t, x_0}\left[w_t\left[\lambda_1\left\|x_0-\hat{x}_0\left(x_t, t, \theta\right)\right\|^2+\lambda_2\left\|\mathcal{F} \mathcal{F} \mathcal{T}\left(x_0\right)-\mathcal{F F} \mathcal{T}\left(\hat{x}_0\left(x_t, t, \theta\right)\right)\right\|^2\right]\right]$$

### 3.4. Conditional Generation for Time Series Applications

- **Conditional extensions of the Diffusion-TS**, in which the modeled $$x_0$$ is conditioned on targets $$y$$​
- 목표는 pre-trained diffusion model과 the gradients of a classifier를 활용하여
  - Posterior $$p\left(x_{0: T} \mid y\right)=\prod_{t=1}^T p\left(x_{t-1} \mid x_t, y\right)$$에서 sampling하는 것
- $$p\left(x_{t-1} \mid x_t, y\right) \propto p\left(x_{t-1} \mid x_t\right) p\left(y \mid x_{t-1}, x_t\right)$$이므로 bayse theorem을 통해 gradient update
  - Score function $$\nabla_{x_{t-1}} \log p\left(x_{t-1} \mid x_t, y\right)=\nabla_{x_{t-1}} \log p\left(x_{t-1} \mid x_t\right)+\nabla_{x_{t-1}} \log p\left(y \mid x_{t-1}\right)$$
  - $$\log p\left(x_{t-1} \mid x_t\right)$$은 diffusion model에서 정의됨.
  - $$\log p\left(y \mid x_{t-1}\right)$$는 classifier에서 parametrize되며, $$\nabla_{x_{t-1}} \log p\left(y \mid x_{0 \mid t-1}\right)$$로 근사됨
- 즉 classifier가 높은 likelihood를 가진 영역에서 sample이 생성되도록 하는 것
  - : $$\tilde{x}_0\left(x_t, t, \theta\right)=\hat{x}_0\left(x_t, t, \theta\right)+\eta \nabla_{x_t}\left(\left\|x_a-\hat{x}_a\left(x_t, t, \theta\right)\right\|_2^2+\gamma \log p\left(x_{t-1} \mid x_t\right)\right)$$
  - where Conditional part $$x_a$$, generative part $$x_b$$
  - gradient term은 reconstruction-based guidance, $$\eta$$로 강도 조절
- 각 diffusion step에서 이 gradient update를 여러 번 반복하여 quality 높인다
- Replacing: $$\tilde{x}_a\left(x_t, t, \theta\right):=\sqrt{\bar{\alpha}_t} x_a+\sqrt{ } 1-\bar{\alpha}_t \epsilon$$을 통해, $$\tilde{x}_0$$를 사용한 sample $$x_{t-1}$$가 생성됨

## 4. Empirical Evaluaiton

### 4.2. Metrics

- Discriminative score (Yoon et al., 2019): measures the similarity using a classification model to distinguish between the original and synthetic data as a supervised task;
- Predictive score (Yoon et al., 2019):  measures the usefulness of the synthesized data by training a post-hoc sequence model to predict next-step temporal vectors using the train-synthesis-and-test-real (TSTR) method;
- Context-Frechet Inception Distance (Context-FID) score ´ (Paul et al., 2022):  quantifies the quality of the synthetic time series samples by computing the difference between representations of time series that fit into the local context; 
- Correlational score (Ni et al., 2020): uses the absolute error between cross correlation matrices by real data and synthetic data to assess the temporal dependency

### 4.3. Interpretability Results

![그림3](/assets/img/timeseries/Diffusion-TS/fig3.png)

- the corrupted samples (shown in (a)) with 50 steps of noise added as input
- outputs the signals (shown in (c)) that try to restore the ground truth (shown in (b))
- with the aid of the decomposition of temporal trend (shown in (d)) and season & error (shown in (e)).
- Result: As would be expected, the trend curve follows the overall shape of the signal, while the season & error oscillates around zero !

### 4.4. Unconditional Time Series Generation

![그림21](/assets/img/timeseries/Diffusion-TS/table1.png)

![그림4](/assets/img/timeseries/Diffusion-TS/fig4.png)

### 4.5. Conditional Time Series Generation

![그림6](/assets/img/timeseries/Diffusion-TS/fig6.png)

### 4.6. Ablaction Study

![그림22](/assets/img/timeseries/Diffusion-TS/table2.png)

## 5. Conclusion

- Diffusion-TS, a DDPM-based method for general time series generation
  - TS-specific loss design and transformer-based deep decomposition architecture
- Unconditional로 훈련된 model이 쉽게 conditional로 확장될 수 있음
  - by combining gradients into the sampling !