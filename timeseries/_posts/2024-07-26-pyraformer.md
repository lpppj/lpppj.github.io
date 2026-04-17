---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2022](https://arxiv.org/pdf/2012.07436)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting (ICLR 2022)

## Abstract

- Pyraformer: a flexible but parsimonious model that can capture a wide range of temporal dependencies, by exploring the multi-resolution representation of the time series
  -  Pyramidal attention module (PAM)
    - the inter-scale tree structure summarizes features at different resolutions
    - the intra-scale neighboring connections model the temporal dependencies of different ranges
  - the maximum length of the signal traversing path in Pyraformer is a constant with regard to the sequence length L (i.e. $$\mathcal{O}(1)$$)

## 1. Introduciton

- 시계열 예측에서 Challenge는 powerful but parsimonious model

![그림1](/assets/img/timeseries/pyraformer/fig1.png)

![그림11](/assets/img/timeseries/pyraformer/table1.png)

- Pyraformer: to simultaneously capture temporal dependencies of different ranges in a compact multi-resolution fashion

## 2. Related Work

pass

## 3. Method

### 3.1. Pyramidal Attention Module (PAM)

- The inter-scale connections form a C-ary tree, in which each parent has C children.
  - the nodes at coarser scales can be regarded as the daily, weekly, and even monthly features of the time series
- $$\to$$ The pyramidal graph offers a multi-resolution representation of the original time series !
  - long-range dependencies 파악이 쉬워짐. 그냥 이웃 노드 연결하기만 하면 되니까 (intra-scale)

- Original Attention mechanism
  - input $$ X$$, output $$ Y$$
  - Query $${Q}={X} {W}_Q$$, key $${K}={X} {W}_K$$, value $${V}={X} {W}_V$$
  - where $${W}_Q, {W}_K, {W}_V \in \mathbb{R}^{L \times D_K}$$
  - Then, $${y}_i=\sum_{\ell=1}^L \frac{\exp \left({q}_i {k}_{\ell}^T / \sqrt{D_K}\right) {v}_{\ell}}{\sum_{\ell=1}^L \exp \left({q}_i {k}_{\ell}^T / \sqrt{D_K}\right)}$$
  -  time and space complexity $$\mathcal{O}(L^2)$$

- Pyramidal Attention Module (PAM)

![그림21](/assets/img/timeseries/pyraformer/fomula2.png)

![그림31](/assets/img/timeseries/pyraformer/myfig1.png)	

- Then, $${y}_i=\sum_{\ell \in \mathbb{N}_{\ell}^{(s)}} \frac{\exp \left({q}_i {k}_{\ell}^T / \sqrt{d_K}\right) {v}_{\ell}}{\sum_{\ell \in \mathbb{N}_l^{(s)}} \exp \left({q}_i {k}_{\ell}^T / \sqrt{d_K}\right)}$$
- 모든 시점끼리 attention을 하지 않고 conv filter로 nodes를 만들고 이웃 노드끼리 attention !

### 3.2. Coarser-saleㄴ Construvtion Module (CSCM)

![그림3](/assets/img/timeseries/pyraformer/fig3.png)

- PAM이 작동할 수 있도록 pyramidal 구조를 initialize하는 역할

### 3.3. Prediction Module

- input embedding 할 때에 예측하고자 하는 길이만큼 붙여서 CSCM, PAM을 통과하면
- 예측 시점에 대한 representation을 얻을 수 있음

![그림33](/assets/img/timeseries/pyraformer/myfig3.png)

## 4. Experiments

![그림12](/assets/img/timeseries/pyraformer/table2.png)

![그림13](/assets/img/timeseries/pyraformer/table3.png)

![그림14](/assets/img/timeseries/pyraformer/fig4.png)

## 5. Conclusion and Outlook

- Pyraformer: a novel model based on pyramidal attention
  - effectively describe both short and long temporal dependencies with low time and space complexity
  -  CSCM to construct a C-ary tree, and then design the PAM to pass messages in both the inter-scale and the intra-scale fashion
  -  Pyraformer can achieve the theoretical $$\mathcal{O}(L)$$ complexity and $$\mathcal{O}(1)$$ maximum signal traversing path length (L: input sequence length)

