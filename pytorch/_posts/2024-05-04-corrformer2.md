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

# (Corrformer, NMI 2023) Code Review 2 - Data Embedding

- step 0. Initialization and Normalizaiton
- step 1. Data Embedding Instance
- step 2. Encoder Instance
- step 3. Decoder Instance
- step 0 ~ 3는 embedding, encoder, decoder를 정의한 것이고, forward에서 실행된다.
  - forward에서 각 step을 지날 때에 step 0 ~ 3을 자세히 보도록 한다.

### step 4. forward
![사진6](/assets/img/pytorch/corrformer1/fig6.png)
- step 1. embedding은 `enc_out = self.enc_embedding(x_enc, x_mark_enc)` 부분
- step 2. encoder는 `enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)` 부분
- step 3. decoder는 `seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)` 부분이다.

### step 0. Initialization and Normalizaiton
![사진0](/assets/img/pytorch/corrformer2/corrformer200.png)
  - 이전 포스팅에서 봤던 `batch_x`가 `x_enc`로 들어간다. (`torch.Size([1, 48, 3850])`)
  - `x_enc`를 normalization해주고 RevIN처럼 learnable parameters로 설정한다.
  - `self.decomp`에서는 seasonal(x - moving average)와 trend(moving average)로 분해한다. (shape은 똑같이 `torch.Size([1, 48, 3850])`)

### step 1. Data Embedding Instance
  - `enc_out = self.enc_embedding(x_enc, x_mark_enc)`
![사진3](/assets/img/pytorch/corrformer2/corrformer23.png)
- `Corrformer.sh`에서 B:1 / L:48 / D:3850 / C:4 / node_num:350 이므로
- `x_enc`는 `1, 48, 3850` --`view`--> `(1, 48, 350, 11)` --`permute`--> `(1, 350, 48, 11)` --`view`--> `(350, 48, 11)`이고
- `x_mark_enc`는  `(1, 48, 4)` --`unsqueeze`--> `(1, 1, 48, 4)` --`repeat`--> `(1, 350, 48, 4)` --`view`--> `(350, 48, 4)`가 된다.
- `self.enc_embedding`의 input이 된다.
![사진0](/assets/img/pytorch/corrformer2/corrformer20.png)
- encoder embedding은 `DataEmbedding`으로 정의되는데 (그리고 decoder embedding도) `Embed.py`에서 import
![사진1](/assets/img/pytorch/corrformer2/corrformer21.png)
- forward의 x는 `value_embedding(x) + temporal_embedding(x_mark) + national_embedding(national_position)`
- 논문에서 아래 부분에 해당한다.
![사진2](/assets/img/pytorch/corrformer2/corrformer22.png)
- GeoPositionalEmbedding, TokenEmbedding, TemporalEmbedding의 코드를 첨부하지는 않겠지만 코드를 보면 각각 national, value, temporal을 `d_model=768` 차원으로 embedding한다.
  - GeoPositionalEmbedding은 1개의 linear layer 사용
  - TokenEmbedding은 channel-wise self-attention + conv1d 사용
  - TemporalEmbedding은 sin과 cos를 사용했다.
  - `self.enc_embedding(x_enc, x_mark_enc)`는 torch.Size([`350, 48, 768`])이다. (`d_model=768`)
    - x_enc       --TokenEmbedding--> `(350, 48, 768)` \\
    $$+$$ x_mark_enc  --TemporalEmbedding--> `(350, 48, 768)` \\
    $$+$$ GeoPositionalEmbedding `(350, 48, 768)`

- step 2부터는 다음 포스팅에서 다루도록 한다.