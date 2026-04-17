---
layout: post
related_posts:
  _
title: 
description: >
  [IJCAI 2023](https://arxiv.org/pdf/2202.07125.pdf)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# (Survey paper) Transformers in Time Series: A Survey (IJCAI 2023)

## Abstract

- Transformer는 long-range dependencies and interactions를 학습할 수 있다.
- 본 논문에서는 Network structure 관점에서 Transformer를 TS forecasting에 사용하기 위해 어떤 adaptaion and modification을 했는지 알아보고
- Application 관점에서 forecasting, anomaly detection, and classification을 포함한 task에 대해 얼마나 잘 작동하는지 알아본다.
- 마지막으로 future direction을 제시한다.

## 1. Introduction

- Transformer : ability for long-range dependencies and interactions in sequential data
- Time series : How to effectively model long-range and short-range temporal dependency and capture seasonality simultaneously ?
- Network modification 관점 : low-level(i.e. module)부터 high-level(i.e. architecture)
- Application 관점 : summarize Transformer for forecasting, anomaly detection, and classification

## 2. Preliminaries of the Transformer

### 2.1. Vanilla Transformer

- Encoder : a multi- head self-attention module and a position-wise feed-forward network
- Decoder : cross-attention models between the multi-head self-attention module and the position-wise feed-forward network

### 2.2. Input Encoding and Positional Encoding

- No recurrence, instead positional encoding
- Absolute Positional Encoding : $$PE(t)_i= \begin{cases}\sin \left(\omega_i t\right) & i \% 2=0 \\ \cos \left(\omega_i t\right) & i \% 2=1\end{cases}$$
  - $$\quad \omega_i$$ is the hand-crafted frequency for each dim
- Relative Positional Encoding : input의 상대적인 위치에 대해 learnable하지만 train에서 본 적 없는, 더 긴 길이의 input에 대해서 확장이 어려움

### 2.3. Multi-head Attention

- scaled dot-product : $$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q K}^{\mathbf{T}}}{\sqrt{D_k}}\right) \mathbf{V}$$

- Multi-head Attention :

  $$\begin{aligned}MultiHeadAttn (\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Concat}\left(\right. head _1, \cdots, head \left._H\right) \mathbf{W}^O \\ \text{where } head _i= Attention \left(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V\right)\end{aligned}$$

### 2.4. Feed-forward and Residual Network

- The feed-forward network(FFN) : $$FFN\left(\mathbf{H}^{\prime}\right)=\operatorname{ReLU}\left(\mathbf{H}^{\prime} \mathbf{W}^1+\mathbf{b}^1\right) \mathbf{W}^2+\mathbf{b}^2$$
  -  $$\mathbf{H}^{\prime}$$ is outputs of previous layer
  - $$\mathbf{W}^1 \in \mathcal{R}^{D_m \times D_f}$$, $$\mathbf{W}^2 \in \mathcal{R}^{D_f \times D_m}, \mathbf{b}^1 \in \mathcal{R}^{D_f}, \mathbf{b}^2 \in \mathcal{R}^{D_m}$$ are trainable parameters

- Residual connection module followed by a layer normalization module : $$\begin{aligned} \mathbf{H}^{\prime} & =\operatorname{LayerNorm}(\operatorname{Self} \operatorname{Attn}(\mathbf{X})+\mathbf{X}), \\ \mathbf{H} & =\operatorname{LayerNorm}\left(FFN\left(\mathbf{H}^{\prime}\right)+\mathbf{H}^{\prime}\right)\end{aligned}$$
  - SelfAttn(.) : self-attention module
  - LayerNorm(.) : the layer normalization operation

## 3. Taxonomy of Transformers in Time Series

![사진1](/assets/img/timeseries/TFinTSsurvey/fig1.png)

## 4. Network Modifications for Time Series

### 4.1. Positional Encoding

- Vanilla Positional Encoding : fixed, hand-crafted
- Learnable Positional Encoding : more flexible and can adapt to spe- cific tasks
- Timestamp Encoding : calendar timestamps (e.g., second, minute, hour, week, month, and year) and special times- tamps (e.g., holidays and events) $$\to$$ additional position encoding

### 4.2. Attention Module

- Self-attention module : FC layer w weights that are dynamically generated based on the pairwise similarity of input patterns
- Memory complexity $$O(N^2)$$
  - explicitly introducing a sparsity bias into the attention mechanism
    - e.g. LogTrans [Li *et al.*, 2019, Pyraformer [Liu *et al.*, 2022a]
  - exploring the low-rank property of the self-attention matrix to speed up the computation,
    - e.g. Informer [Zhou *et al.*, 2021], FEDformer [Zhou *et al.*, 2022].

![사진2](/assets/img/timeseries/TFinTSsurvey/table1.png)

### 4.3. Architecture-based Attention Innovation

- Hierarchical architecture : multi-resolution aspect of TS
  - Informer [Zhou *et al.*, 2021]
    - max-pooling layers with stride 2 btw attention blocks (down-sample series into its half slice)
  - Pyraformer [Liu *et al.*, 2022a]
    - C-ary tree-based attention mechanism (finest-origin / coarser-lower resolutions)
    -  both intra-scale and inter-scale attentions $$\to$$ temporal dependencies across different resolutions

## 5. Applications of Time Series Transformers

![사진9](/assets/img/timeseries/TFinTSsurvey/fig2.jpeg)

### 5.1. Transformers in Forecasting

- **Module-level variants** : main architectures는 비슷한데, minor changes
  - (1) designing new attention modules
    - LogTrans [Li *et al.*, 2019] : convolution self-attention, sparse bias (Logsparse mask)
    - Informer [Zhou *et al.*, 2021] : selects dominant queries based on queries and key similarities
    -  AST [Wu *et al.*, 2020a] : generative adversarial encoder- decoder framework to train a sparse Transformer
    -  Pyraformer [Liu *et al.*, 2022a] : hierarchical pyramidal attention module with a binary tree following the path
    - Quatformer [Chen *et al.*, 2022] : learning-to-rotate attention (LRA) based on quaternions that introduce learnable period and phase information
    - FEDformer [Zhou *et al.*, 2022] : attention operation in the frequency domain with Fourier trans- form and wavelet transform
  - (2) exploring the innovative way to normalize time series data
    -  Non-stationary Transformer [Liu *et al.*, 2022b]
  - (3) utilizing the bias for token inputs
    - Autoformer [Wu *et al.*, 2021] : segmentation-based representation mechanism (auto-correlation mechanism)
    - PatchTST [Nie *et al.*, 2023] : subseries-level patch design which are served as input tokens w/ channel-independency
    - Cross- former [Zhang and Yan, 2023] : input is embedded into a 2D vector array and then two-stage attention layer is used to efficiently capture the cross-time and cross-dimension dependency

- **Architecture-level variants**
  - Triformer [Cirstea *et al.*, 2022] : triangular,variable-specific patch attention $$\to$$ lightweight and linear complex- ity
  - Scaleformer [Shabani *et al.*, 2023] : iteratively refine the forecasted time series at multiple scales with shared weights.

- **Spatio-temporal Forecasting, Evnet Forecasting**
  - Pass

### 5.2. Transformers in Anomaly Detection

- Transformer + Generative models

  - TranAD [Tuli *et al.*, 2022] : Transformer는 small deviation of anomaly는 놓치므로 reconstruction errors를 amplify하는 adversarial training

  - MT-RVAE [Wang *et al.*, 2022], TransAnomaly [Zhang *et al.*, 2021] : Transformer + VAE $$\to$$ reduce training costs, 다양한 scale의 정보 통합

### 5.3 Transformers in Classification

- GTN [Liu *et al.*, 2021] : two-tower Transformer(time-step-wise attention and channel-wise attention) $$\to$$  learnable weighted concatenation
- TARNet [Chowdhury *et al.*, 2022] : utilizes attention score for important timestamps masking and reconstruction

## 6. Experimental Evaluation and Discussion

- **Robustness Analysis**
  - 대부분의 attention-based models는 lower the quadratic calculation and memory complexity를 위해 module을 수정했고, 좋은 실험 결과를 위해 짧은 input을 사용했는데, 긴 input을 넣어도 MSE가 커지지 않고 잘 유지되는지 확인했다.
  - 대부분의 모델들이 긴 input에 대해서는 잘 처리하지 못한다.

![사진3](/assets/img/timeseries/TFinTSsurvey/table2.png)

- **Model Size Analysis**
  - 일반적으로 model size가 커지면 prediction power도 좋아지는데 attention-based models에서 그렇지 않음을 확인했다.
  - 지금까지의 Transformer 자체가 features를 잘 뽑아내지 못하는 구조일 수 있겠다.

![사진4](/assets/img/timeseries/TFinTSsurvey/table3.png)

- **Seasonal-Trend Decomposition Analysis**
  - seasonal-trend decomposition는 Transformer에서 필수적인 부분 : model performance가 50% - 80% boosting

![사진5](/assets/img/timeseries/TFinTSsurvey/table4.png)

## 7. Future Research Opportunities

### 7.1. Inductive Biases for Time Series Transformers

- Channel-independence와 Cross-channel(dim) dependency은 서로 반대 inductive bias이지만 둘 다 실험 결과가 좋았다.
- 즉 cross-channel learning에는 noise도 있고 signal도 있다는 의미
- 어떤 inductive bias를 어떻게 induce할지 고려할 필요가 있음

### 7.2. Transformers and GNN for Time Series

- Traffic forecsting처럼 spatial dependency (relationship among dim)이 강한 경우에는 GNN + Transformer의 성능이 좋을 수 있다.

### 7.3. Pre-trained Transformers for Time Series

- Large-scale pre-trained Transformer model의 성능이 좋긴 한데 대부분 classification을 위한 pre-train이라는 점에서, 다른 tasks를 위한 pre-train도 고려할 수 있다.

### 7.4. Transformers with Architecture Level Variants

- 지금까지는 attention module에 대한 modification이 주로 등장했지만, TS를 위한 architecture-level design도 고려할 수 있다.

### 7.5. Transformers with NAS for Time Series

- Neural architecture search(NAS)과 같은 AutoML을 통해, 성능에 영향을 주는 embedding dimension이나 head/layer의 개수 등 효율적인 architecture를 고려할 수 있다.

## 8. Conclusion

- new taxonomy consisting of network design and application
- strengths and limitations of representative methods by experimental evaluation
- highlight future research directions.