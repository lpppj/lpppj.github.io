---
layout: post
related_posts:
  _
title: 
description: >
  [Philos Trans R Soc A. 2020](https://arxiv.org/pdf/2004.13408.pdf)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# (Survey paper) Time Series Forecasting With Deep Learning A Survey (Philos Trans R Soc A. 2020)

## 1. Introduction
- Time series forecasting
  - traditional methods : parametric models informed by domain expertise (ex. autoregressive(AR))
  - machine learning : learn temporal dynamics in a puerly data-driven manner
  - deep learning : learn complex data representation
    - CNN, RNN, Attention-based mechanism
  - hybrid model : Quantitative TS model + deep learning model
- Time series forecasting의 application
  - interpretability and counterfactual prediction

## 2. Deep Learning Architectures for TSF
- One-step-ahead forecasting : $$\hat{y}_{i, t+1}=f\left(y_{i, t-k: t}, \boldsymbol{x}_{i, t-k: t}, \boldsymbol{s}_i\right)$$
  - $$\hat{y}_{i, t+1}$$ : model forecast
  - $$y_{i, t-k: t}=\left\{y_{i, t-k}, \ldots, y_{i, t}\right\}, \boldsymbol{x}_{i, t-k: t}=\left\{\boldsymbol{x}_{i, t-k}, \ldots, \boldsymbol{x}_{i, t}\right\}$$ : observation over look-back window
  - $$f(\cdot)$$ : the prediction function learntby the model
### 2.(a) Basic building blocks
- Encoder : $$\boldsymbol{z}_t=g_{\mathrm{enc}}\left(y_{t-k: t}, \boldsymbol{x}_{t-k: t}, \boldsymbol{s}\right)$$
- Decoder : $$f\left(y_{t-k: t}, \boldsymbol{x}_{t-k: t}, \boldsymbol{s}\right)=g_{\mathrm{dec}}\left(\boldsymbol{z}_t\right)$$
  - Encoder에서 observations를 latent vector로 representation
  - (1) Convolution Neural Networks : $$\begin{aligned} \boldsymbol{h}_t^{l+1} & =A((\boldsymbol{W} * \boldsymbol{h})(l, t)) \\ (\boldsymbol{W} * \boldsymbol{h})(l, t) & =\sum_{\tau=0}^k \boldsymbol{W}(l, \tau) \boldsymbol{h}_{t-\tau}^l \end{aligned}$$
    - Convolution과 pooling을 반복하는 구조. TS에서는 과거의 값만 보도록 설계
    - $$\boldsymbol{h}_t^l \in \mathbb{R}^{\mathcal{H}_{i n}}$$ : intermediate state at layer $$l$$ at time $$t$$
    - $$*$$ : convolution operator
  
    - $$\boldsymbol{W}(l, \tau) \in$ $\mathbb{R}^{\mathcal{H}_{\text {out }} \times \mathcal{H}_{\text {in }}}$$ : fixed filter weight at layer $$l$$
    - $$A(.)$$ : activation function
  - Dilated Convolution : $$(\boldsymbol{W} * \boldsymbol{h})\left(l, t, d_l\right)=\sum_{\tau=0}^{\left\lfloor k / d_l\right\rfloor} \boldsymbol{W}(l, \tau) \boldsymbol{h}_{t-d_l \tau}^l$$
    - $$d_l$$ : layer-specific dilation rate
    - (WaveNet) $$d_l = 2^l$$ at layer $$l$$ (fig1.(a))
    ![사진1](/assets/img/timeseries/TSwDLsurvey/fig1.png)
  - (2) Recurrent Neural Networks
    - Memory state를 통해 과거 정보를 기억하는 sequential data에 적합한 구조
    - Gradient vanishing으로 인한 long-range dependency $$\to$$ LSTM

    - Memory update funciton : $$\boldsymbol{z}_t=\nu\left(\boldsymbol{z}_{t-1}, y_t, \boldsymbol{x}_t, \boldsymbol{s}\right)$$


    - Network : $$\begin{aligned} y_{t+1} & =\gamma_y\left(\boldsymbol{W}_y \boldsymbol{z}_t+\boldsymbol{b}_y\right) \\ \boldsymbol{z}_t & =\gamma_z\left(\boldsymbol{W}_{z_1} \boldsymbol{z}_{t-1}+\boldsymbol{W}_{z_2} y_t+\boldsymbol{W}_{z_3} \boldsymbol{x}_t+\boldsymbol{W}_{z_4} \boldsymbol{s}+\boldsymbol{b}_z\right) \end{aligned}$$
      - $$W_{.}, \boldsymbol{b}$$ : the linear weights and bias
      - $$\gamma_y(.), \gamma_z(.)$$ : network activation functions
    - Long Short Term Memory(LSTM)
    ![사진2](/assets/img/timeseries/TSwDLsurvey/fig2.png)
  - (3) Attention mechanisms
    - form : $$\boldsymbol{h}_t=\sum_{\tau=0}^k \alpha\left(\boldsymbol{\kappa}_t, \boldsymbol{q}_\tau\right) \boldsymbol{v}_{t-\tau}$$
      - key $$\boldsymbol{\kappa}_t$$, query $$\boldsymbol{q}_\tau$$ and value $$\boldsymbol{v}_{t-\tau}$$ are intermediate features produced at different time steps by lower levels of the network
      - $$\alpha\left(\boldsymbol{\kappa}_t, \boldsymbol{q}_\tau\right) \in[0,1]$$ is the attention weight for $$t-\tau$$ generated at time $$t$$
      - $$\boldsymbol{h}_t$$ is the context vector output of the attention layer
### 2.(b) Multi-horizon Forecasting Models
- 단순히 다음 한 시점에 대한 예측이 아닌 미래 여러 시점에 대한 예측
- (1) Iterative Methods : Autoregressive forecasting. 각 time step에서의 작은 오차가 누적된다는 단점이 있다.
- (2) Direct Methods : Encoder의 정보를 활용해서 한 번에 target time steps를 예측

## 3. Incorporating Domain Knowledge with Hybrid Models
- Machine learning의 underperformance의 이유는 1) flexibility로 인한 overfitting, 2) pre-processed input에 대한 sensitivity
- Hybrid models
  - combine well-studied quantitative time series models together with deep learning
  - use domain knowledge $$\to$$ hypothesis space를 줄여준다
  - (a) Non-probabilistic Hybrid models : forecasting equations를 modify
  - (b) Probabilistic Hybrid models : predictive distribution으로 parameters 생성
  
## 4. Facilitating Decision Support Using Deep Neural Networks
- 연구하는 입장에서는 model의 성능(MSE, Accuracy, ...)가 중요하지만, user는 future action에 대한 guide의 지표
- 그러므로 Local Interpretable Model-Agnostic Explanations (LIME), Shapley additive explanations (SHAP)과 같은 post-hoc 분석, Attention weights를 통한 inherent interpretability를 이해할 필요가 있다.
- 그러면 counterfactual forecast(determining what would have happened if a different set of circumstances had occurred) 가능

## 5. Conclusions and Future Directions
- Survey the main architectures used for TS forecasting
- Hybrid DL models : combine statistical and deep learning components
- Limitation : irregular TS나 hierarchical structure에 대한 고민은 하기 이전

## 추가
- 2020년에 발표된 survey 논문이지만 최근 Long-term Time Series Forecasting(LTSF)에 활용되는 모델에 대한 내용을 잘 정리한 논문이다.
- 본 논문 이후 현재까지 최신 연구들을 이해한 상태로 읽는다면 최신 연구들의 motivation을 이해하는 데에 도움이 되는 논문이다.