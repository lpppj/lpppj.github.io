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

# Chernoff Bound (upper bound on the tail probability)

## Prerequirement
- Markov inequality $$\mathrm{P}(X \geq a) \leq \frac{\mathrm{E}(X)}{a}$$
- Chebyshev inequality $$\operatorname{Pr}(\mid X-\mu\mid \geq k \sigma) \leq \frac{1}{k^2}$$

## Chernoff Bound
- 확률변수 X에 대한 tail probability \\
  $$\begin{aligned}P(X \geqslant a) & =P\left(e^{t X} \geqslant e^{t a}\right) \\ & \leqslant \frac{\mathbb{E}\left[e^{t X}\right]}{e^{t a}} \text { by Markov inequality } . \\ & =M(t) e^{-t a}\end{aligned}$$

## Example : Bernoulli Confidence sets vs. Hoeffding's inequality
- Under asymptotic normality \\
  $$\begin{aligned}P_\theta\left(\theta \in C_n\right)&=P_\theta\left(\widehat{\theta}-z_{\alpha / 2} \sqrt{\operatorname{Var}_\theta[\widehat{\theta}]} \leq \theta \leq \widehat{\theta}+z_{\alpha / 2} \sqrt{\operatorname{Var}_\theta[\widehat{\theta}]}\right)\\C_n&=\left(\widehat{p}-z_{\alpha / 2} \sqrt{\frac{\widehat{p}(1-\widehat{p})}{n}}, \widehat{p}+z_{\alpha / 2} \sqrt{\frac{\widehat{p}(1-\widehat{p})}{n}}\right)\end{aligned}$$
- Hoeffding's inequality \\
  $$P(\mid \widehat{p}-p\mid \geq t) \leq 2 e^{-2 n t^2}$$
- **Under asymptotic normality로 얻은 Confidence sets은 항상 Hoeffding's inequality 보다 tight(short)하다.**
  - **증명** :  $$z_{\alpha / 2} \sqrt{\frac{\widehat{p}(1-\widehat{p})}{n}} \leq \sqrt{\frac{\log (2 / \alpha)}{2 n}}$$에서 $$\widehat{p}(1-\widehat{p}) \le \frac{1}{4}$$ 이므로
  - (1) Exponential Markov Inequality\\
    $$\begin{aligned} P(z \geqslant z) & =P\left(e^{t Z} e^{t z}\right) \\ & \leqslant \frac{\mathbb{E}\left[e^{t Z}\right]}{e^{t z}}=e^{\frac{t^2}{2}-t z} \\ & \leqslant e^{-z^2 / 2} \quad \text{ when }t=z\end{aligned}$$
  - (2) Chernoff Bounds\\
    $$\begin{aligned}  p(Z \geqslant z) &\leqslant e^{-z^2 / 2} \\ p\left(z \geqslant z_{\frac{\alpha}{2}}\right) &\leqslant e^{-z_{\frac{\alpha}{2}}^2 / 2} \\  \alpha / 2 &\leqslant e^{-z_{\frac{\alpha}{2}}^2 / 2} \\ \log{\frac{\alpha}{2}}&\leqslant-\frac{z_{\frac{\alpha}{2}}^2}{2} \\ z_{\frac{\alpha}{2}}^2 &\leq 2 \log{\frac{\alpha}{2}} \\ \end{aligned}$$