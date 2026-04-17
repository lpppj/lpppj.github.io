---
layout: post
related_posts:
  _
title: 
description: >
  [Corrformer github](https://github.com/thuml/Corrformer)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# (Corrformer, NMI 2023) Code Review 3 - Encoder

![사진10](/assets/img/pytorch/corrformer0/corrformer010.jpeg)
- 우리는 `exp_main`의 `train` 메소드를 실행하고 있다.
- `get_data`는 이미 살펴보았고 `self.model`에 들어가는 Corrformer를 이해하기 위해 `Corrformer.py`를 살펴보고 있다.
  - step 0. Initialization and Normalizaiton
  - step 1. Data Embedding Instance
  - step 2. Encoder Instance
  - step 3. Decoder Instance
  - step 4. forward

- forward 전체 코드
![사진6](/assets/img/pytorch/corrformer1/fig6.png)

### step 2. Encoder Instance
  - `enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)`
![사진3](/assets/img/pytorch/corrformer2/corrformer23.png)
- 지난 포스팅에서 봤던 `enc_out = self.enc_embedding(x_enc, x_mark_enc) #(350, 48, 768)`가 `encoder`의 input
- `encoder`는 `layers'에 있는 Encoder, MultiCorrelation, AutoCorrelation, CrossCorrelation, CausalConv를 import
![사진31](/assets/img/pytorch/corrformer3/corrformer31.png)

- `self.encoder` instance를 만드는데, parameters를 잠시 지우고 구조만 보면 아래와 같다. 

~~~python
self.encoder = Encoder([
        EncoderLayer(
            MultiCorrelation(
                AutoCorrelationLayer(AutoCorrelation()),
                CrossCorrelationLayer(CrossCorrelation(CausalConv()))
            ))])
~~~

- `Encoder` 안에 `EncoderLayer`가 있다. `EncoderLayer` 안에는 `MultiCorrelation`이 있는데, `MultiCorrelation` 안에는 `AutoCorrelationLayer(AutoCorrelation())`과 `CrossCorrelationLayer(CrossCorrelation(CausalConv()))`이 있다.

### CausalConv()

~~~python
# Corrformer.py
CausalConv(
    num_inputs=configs.d_model // configs.n_heads * (self.label_len + self.pred_len),
    num_channels=[configs.d_model // configs.n_heads * (self.label_len + self.pred_len)] * configs.dec_tcn_layers,
    kernel_size=3)

# Corrformer.sh
# --d_model 768 \ --n_heads 16 \ --label_len 24 \ --pred_len 24 \ --dec_tcn_layers 1 \

# Causal_Conv.py
class CausalConv(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(CausalConv, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 3 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [CausalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
~~~

