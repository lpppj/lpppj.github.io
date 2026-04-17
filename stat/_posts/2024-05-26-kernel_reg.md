---
layout: post
related_posts:
  _
title: 
description: >
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# Analysis of non-parametric kernel regression

- 일반적인 regression은 response variable Y와 covariate X의 관계를 이해하는 방법 중 하나로 일종의 conditional expectation이다. \\
  $$ r(x)=\mathbb{E}[Y \mid X=x]=\int y f(y \mid x) dy $$
- non-parametric regession을 하는 기본적인 방법 중 하나는 kernel regression이다. \\
  estimator는 $$\widehat{r}(x)=\sum_{i=1}^n w_i(x) Y_i$$ where $$w_i(x)=\frac{K\left(\frac{x-X_i}{h}\right)}{\sum_{j=1}^n K\left(\frac{x-X_j}{h}\right)}$$
  - weights는 x 주변에 높은 가중치를 두고, h는 smoothing을 결정하는 bandwidth이다.
  - 간단한 kernel의 예시로 Gaussian kernel을 사용한다. $$K(x)=\frac{1}{\sqrt{2 \pi}} \exp \left(-x^2 / 2\right) $$

## Assumptions
- 증명을 복잡하지 않게 하기 위한 가정 $$y_i=r\left(x_i\right)+\epsilon_i$$, where
  - **Design Assumption** : $$x_i$$ is one-dimensional, and **equally spaced** on $$[0,1]$$
    - 사실 꼭 필요한 가정은 아니지만 특정 x 근처에 다른 x가 존재함을 가정하기 위한 가정이다.
  - **Regression function Assumption** : $$r(x)=\mathbb{E}[Y \mid X=x] \text { is } L \text {-Lipschitz}$$ i.e. $$\left\vert\frac{d}{d x} r(x)\right\vert \leq L$$
  - **Noise Assumption** : $$\mathbb{E}\left[\epsilon_i\right]=0, \operatorname{Var}\left[\epsilon_i\right]=\sigma^2$$ and iid
  - **Kernel Assumption** : spherical kernel $$K(x)=\mathbb{1}(-1 \leq x \leq 1)$$

## Kernel Regression
- Under the **Assumptions** with $$h \ge 1/(n-1)$$, \\
  $$\begin{aligned} R(\widehat{r}, r) & = MSE(\widehat{r}, r) \\ & =\int_0^1\{\widehat{r}(x)-r(x)\}^2 d x \\ & =\int_0^1 \operatorname{bias}^2(x) d x+\int_0^1 \operatorname{Var}(\widehat{r}(x)) d x \leq L^2 h^2+\frac{\sigma^2}{(n-1) h} \end{aligned}$$
- Proof : 적분 구간이 [0,1]이므로 $$\max _{x \in[0,1]}\left\vert\operatorname{bias}(x)\right\vert \leq L h$$, 그리고 $$\max _{x \in[0,1]} \operatorname{Var}(\widehat{r}(x)) \leq \frac{\sigma^2}{(n-1) h}$$를 증명
  - **Bounding the bias** \\
    $$\begin{aligned} \left\vert \operatorname{bias}(x)\right\vert=\left\vert \mathbb{E} \widehat{r}(x)-r(x)\right\vert & \stackrel{\text { (i) }}{=}\left\vert \mathbb{E}\left[\sum_{i=1}^n\left(w_i(x)\left(Y_i-r(x)\right)\right)\right]\right\vert \\ &  \stackrel{\text { (ii) }}{=} \left\vert\sum_{i=1}^n\left(w_i(x)\left(r\left(X_i\right)-r(x)\right)\right)\right\vert \\ & \stackrel{\text { (iii) }}{\leq} \sum_{i=1}^n w_i(x)\left\vert r\left(X_i\right)-r(x)\right\vert \\ & \stackrel{\text { (iv) }}{\leq} L h \sum_{i=1}^n w_i(x)=L h,\end{aligned}$$ \\
    (i) is holds since $$\sum_{i=1}^n w_i(x)=\frac{\sum_{i=1}^n K\left(\frac{x-X_i}{h}\right)}{\sum_{j=1}^n K\left(\frac{x-X_j}{h}\right)}=1$$ \\
    (ii) follows by $$\mathbb{E}\left[Y_i \mid X_i\right]=r\left(X_i\right)$$ \\
    (iii) uses triangle inequality \\
    (iv) if $$ \left\vert X_i-x\right\vert \leq h,$$ then $$\left\vert r\left(X_i\right)-r(x)\right\vert \leq Lh$$ by the Lipschitz  and if $$\left\vert X_i-x\right\vert>h,$$ then $$w_i(x)=0$$
  - **Bounding the variance**
    - 먼저 weights에 대한 bound $$w_i(x)=\frac{K\left(\frac{x-X_i}{h}\right)}{\sum_{j=1}^n K\left(\frac{x-X_j}{h}\right)}=\frac{\mathbb{1}\left(\left\vert x-X_i\right\vert \leq h\right)}{\sum_{j=1}^n \mathbb{1}\left(\left\vert x-X_j\right\Vert \leq h\right)} \leq \frac{1}{(n-1) h}$$를 보인다. 
    - 즉 $$\sum_{j=1}^n \mathbb{1}\left(\left\vert x-X_j\right\vert \leq h\right) \geq (n-1)h$$를 보인다.
      - 모든 $$h \geq 1 /(n-1)$$에 대해 $$\min _{x \in[0,1]} \sum_{j=1}^n \mathbb{1}\left(\left\vert x-X_j\right\vert \leq h\right)=\sum_{j=1}^n \mathbb{1}\left(\left\vert X_1-X_j\right\vert \leq h\right)$$이므로 (sum is minimized at the boundary)
      - $$h \in\left[\frac{k}{n-1}, \frac{k+1}{n-1}\right)$$의 경우에 대해 \\
        $$\\ \begin{aligned} \min _{x \in[0,1]} \sum_{j=1}^n \mathbb{1}\left(\left\vert x-X_j\right\vert \leq h\right) & =\sum_{j=1}^n \mathbb{1}\left(\left\vert X_1-X_j\right\vert \leq h\right) \\ & \geq \sum_{j=1}^n \mathbb{1}\left(\left\vert X_1-X_j\right\vert \leq \frac{k}{n-1}\right)=k+1 \\ & \geq(n-1) h \end{aligned}$$
      - 그러므로 \\
        $$\\ \begin{aligned} \operatorname{Var}(\widehat{r}(x)) & =\mathbb{E}\left[(\widehat{r}(x)-\mathbb{E}(\widehat{r}(x)))^2\right]=\mathbb{E}[(\sum_{i=1}^n(w_i(x) \underbrace{\left(Y_i-r\left(X_i\right)\right.}_{=\epsilon_i})))^2] \\ & =\mathbb{E}\left[\left(\sum_{i=1}^n \epsilon_i w_i(x)\right)^2\right] \\ & =\sum_{i=1}^n w_i(x)^2 \mathbb{E}\left(\epsilon_i^2\right)=\sigma^2 \sum_{i=1}^n w_i(x)^2 \\ & \leq \sigma^2 \max _{1 \leq i \leq n} w_i(x) \sum_{i=1}^n w_i(x) \leq \frac{\sigma^2}{(n-1) h} \end{aligned}$$
- Proof : 적분 구간이 [0,1]이므로 $$\max _{x \in[0,1]}\left\vert\operatorname{bias}(x)\right\vert \leq L h$$, 그리고 $$\max _{x \in[0,1]} \operatorname{Var}(\widehat{r}(x)) \leq \frac{\sigma^2}{(n-1) h}$$이 증명되었다.