---
layout: post
related_posts:
  _
title: 
description: >
  [Pathformer github](https://github.com/decisionintelligence/pathformer)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# (Code Review, ICLR 2024) Pathformer

[(Paper) Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting](https://openreview.net/pdf?id=lJkOCMP2aW)

[(Paper Review, ICLR 2024) Pathformer](https://lpppj.github.io/timeseries/2024-05-23-Pathformer)

## 1. Git clone

![사진1](/assets/img/pytorch/pathformer_code/fig1.png)

![사진1](/assets/img/pytorch/pathformer_code/fig2.png)

- 먼저 터미널에 git clone과 requirements를 입력하여 install 한다.
- `bash scripts/multivariate/ETTm2.sh`로 ETTm2 데이터셋을 예측할 수 있다.

## 2. .sh

![사진1](/assets/img/pytorch/pathformer_code/fig3.png)

- `ETTm2.sh` 파일에는 `run.py`를 실행하도록 되어있다.

## 3. run.py

![사진1](/assets/img/pytorch/pathformer_code/fig4.png)

- parser를 통해 arguments를 만든다.

![사진1](/assets/img/pytorch/pathformer_code/fig5.png)

- 그리고 `Exp_Main`에 있는 train에 arguments를 넣어준다.

## 4. exp_main.py

### 4.1. train

![사진1](/assets/img/pytorch/pathformer_code/fig6.png)

- model, data, optimizer, criterion을 설정하는 간단한 함수들과, `vali`, `train`, `test`, `predict` 함수가 있다.

![사진1](/assets/img/pytorch/pathformer_code/fig7.png)

- `Exp_main` : `train`
  - `_get_data`로 train, valid, test 데이터셋을 load
  - `sum(p.numel() for p in self.model.parameters())`는 parameters 개수
  - time, early sipping, optimizer, criterion, learning rate scheduler 정의
  - `lr_scheduler.OneCycleLR`는 learning rate를 빠르게 최대 학습률까지 증가시켰다가 다시 감소시키면서 최적화 과정
    - `optimizer` : 사용하는 optimizer
    - `steps_per_epoch` : 1 epoch가 몇 번의 update가 발생하는지 (mini-batch)
    - `pct_start` : learning rate가 증가하는 구간의 비율 / `epochs` : 전체 epoch 수
    - `max_lr` : 최대 learning rate

![사진1](/assets/img/pytorch/pathformer_code/fig8.png)

- each epoch에서는 train loder에서 batch 단위로 데이터를 받고

![사진1](/assets/img/pytorch/pathformer_code/fig9.png)

- `with torch.cuda.amp.autocast():`는 `float16`과 `float32`를 자동으로 캐스팅
- 모델이 예측한 `outputs`와 정답 `batch_y`를 비교하여 loss 계산

![사진1](/assets/img/pytorch/pathformer_code/fig10.png)

- epoch에 걸린 시간과 loss를 출력하고 backward로 parameters를 update

![사진1](/assets/img/pytorch/pathformer_code/fig11.png)

- Validation set에 대한 loss로 early stopping 여부를 결정하고 학습이 종료되면 모델 저장
- vali 함수는 특이 사항 없으므로 pass

### 4.2. test

![사진1](/assets/img/pytorch/pathformer_code/fig12.png)

- test dataset과, 학습되어 저장된 model을 load한다.

![사진1](/assets/img/pytorch/pathformer_code/fig13.png)

- train과 비슷하게 batch 단위로 모델에 넣어서 예측값을 얻는다.

![사진1](/assets/img/pytorch/pathformer_code/fig14.png)

- batch 20개마다 묶어서 visualizaiton을 한다.

![사진1](/assets/img/pytorch/pathformer_code/fig15.png)

- 최종적인 예측과 loss를 `results.txt`에 저장한다.

### 4.3. predict

pass

## 5. models/Pathformer.py

- `from models import PathFormer` 이므로 해당 경로로 가서 pathformer의 archtecture를 보자

![사진1](/assets/img/pytorch/pathformer_code/fig16.png)

- `forward`는 normalization $$\to$$ `start_fc` $$\to$$ for `layer` in `self.AMS_lists` $$\to$$ de-normalization로 구성된다.
- `forward`에 들어온 x의 shape은 `torch.size([batch_size, seq_len, num_nodes])`이다.
  - `seq_len`은 관측하는 과거 시점 수, `num_nodes`는 multivariate에서 variates 개수
- x가 unsqueeze되어 normalization, `start_fc`를 통과하면 `torch.size([batch_size, seq_len, num_nodes, d_model])`이 된다. (아래 `__init__` 참고)
- 이제 `AMS_list`의 `layers`를 통과하고 denormalization을 통과한다.

![사진1](/assets/img/pytorch/pathformer_code/fig17.png)

- `__init__`을 보면 `self.AMS_lists`는 `layers.AMS`에서 import한다.
- AMS layer가 pathformer는 전부이니 살펴보자

## 6. Layers/AMS.py

![사진1](/assets/img/pytorch/pathformer_code/fig18.png)

- `self.seasonality_and_trend_decompose`
- `self.noisy_top_k_gating`
- `self.cv_squared`
- `SparseDispatcher`와 `SparseDispatcher.dispatch`, `SparseDispatcher.combine`
- `self.experts`
- 각각에 대해서 하나씩 살펴보도록 한다.

### 6.1. self.seasonality_and_trend_decompose

![사진1](/assets/img/pytorch/pathformer_code/fig19.png)

![사진1](/assets/img/pytorch/pathformer_code/fig20.png)

- AMS class 안에서 정의된 함수
- **seasonality와 trend를 x에서 각각 계산**하기 때문에 $$seasonal + trend = x$$가 아님
  - 해당 함수의 결과는 x에 seasonality와 trend를 더한 결과이다.
- 처음에 `x = x[:, :, :, 0]`은 `d_model` 차원으로 표현된 x에서 첫 번째 dimension만 사용해서 decompose한다는 의미

![사진1](/assets/img/pytorch/pathformer_code/fig21.png)

- seasonality_model은 `FourierLayer`
  - 푸리에 변환(fft) 후 amplitude가 높은 frequency $$k$$​​개를 inverse 푸리에 변환(extrapolate)
- trend_model은 `series_decomp_multi`
  - 다양한 크기의 kernel size로 moving average를 softmax

### 6.2. self.noisy_top_k_gating

![사진1](/assets/img/pytorch/pathformer_code/fig22.png)
![사진1](/assets/img/pytorch/pathformer_code/fig23.png)

- `start_linear.squeeze`와 `w_gate`로 `torch.Size([batch, seq_len, num_node])`가 `torch.Size([batch, num_expert])`가 된다.
- 같은 크기 `torch.Size([batch, num_expert])`의 `logit`을 만들고 $$top-k$$ logit을 `gates`에 넣는다.
  - `scatter`는 특정 인덱스 위치에 값을 할당하는 함수이다.
  - `gate`의 shape은 `torch.Size([batch, num_experts])`가 되는데, 각 행(batch)에서 k개를 제외하고는 다 0이다.
  - 그리고 각 행(batch)마다 그 k개가 어떤 experts인지는 다르다.
- `load`는 각 expert가 배치 전체에서 얼마나 선택되었는지에 대한 비율을 의미한다.
  - shape은 `torch.Size([num_experts])`이다.

### 6.3. self.cv_squared

![사진1](/assets/img/pytorch/pathformer_code/fig18.png)

- 다시 AMS.forward로 돌아오자
- 각 expert마다 모든 배치에 대해 sum을 해서 `importance`를 계산하면 `num_experts`개의 숫자가 된다.
- `cv_squared`를 통해 `num_experts`개의 숫자의 변동계수를 구해서 `balance_loss`를 구한다.
  - 변동계수는 $$\frac{\sigma^2}{\mu^2}$$이다.
  - 이 값이 크면 특정 experts에 importance가 몰려있음을 의미한다.

![사진1](/assets/img/pytorch/pathformer_code/fig24.png)

### 6.4. SparseDispatcher (*어려움 주의)

![사진1](/assets/img/pytorch/pathformer_code/fig25.png)
![사진1](/assets/img/pytorch/pathformer_code/fig26.png)

- `__init__`에서 준비해놓는 것들이 많으니 하나하나 보도록 한다. $$k=2$$, `num_experts`=4인 경우이다.

![사진1](/assets/img/pytorch/pathformer_code/fig27.png)
- 각 행은 batch를 의미하기 때문에 행의 개수는 batch size (여기선 512)
- 각 행에는 `num_experts`개의 숫자가 있고 그 중 $$k$$개만 non-negative, 나머지는 0
- 첫 행에서 2, 3번째 숫자가 양수라는 것은, 첫번째 배치에서 2, 3번째 experts가 선택되었다는 것을 의미
- 바로 아래에 있는 `torch.nonzero(gates)`에서도 그 사실을 알 수 있다.
  - 첫 번째 배치에서는 index 1, 2인 experts가, 마지막 배치에서는 index 1, 2인 experts가 선택됨

![사진1](/assets/img/pytorch/pathformer_code/fig28.png)
- 이제 sort를 하는데 첫번째 열은 어차피 index라서 정렬되어있고
  - (두 번째 열이 정렬되면서 섞이기 때문에 두 번째 열의 숫자가 첫 번째 열의 배치 index와 상관 없게 된다)
- 그리고 `index_sorted_experts`는 정렬된 숫자가 몇 번째 index에 있던 숫자인지를 표시해준다.
  - (여기서부터 헷갈리기 시작함)

![사진1](/assets/img/pytorch/pathformer_code/fig29.png)
- `self._expert_index`는 각 배치에서 선택된 experts의 index를 정렬한 것이다.
  - 각 배치마다 $$k$$개씩 있으니 총 batch_size $$\times k$$개의 숫자겠다.
- 그리고 그걸 다시 batch index로 되돌릴 수가 있을 것이다.
  - 즉 `self._batch_index`가 1, 3, 6,...이라는 것은 expert 0이 선택되었던 batch가 1, 3, ...이고
  - 그 다음 expert 1이 선택된 batch들이 몇 번째 batch인지 쭉 나열이 된다. (이걸 마지막 expert까지 반복)
- 마지막으로 `self._part_sizes`는 모든 batches 통틀어서 각 expert가 몇 번 선택되었는지를 의미한다.

![사진1](/assets/img/pytorch/pathformer_code/fig30.png)
- 이제 `gates_exp`는 expert 0이 선택되었던 batches를 쭉 나열하고, 그 다음에 expert 1이 선택되었던 batches를 쭉 나열하고... 마지막 expert가 선택되었던 batches까지 나열한 것이다.
- 그리고 `self._nonzero_gates`는 expert $$i$$ ($$i = 1, ...,$$ `num_experts`)가 선택된 배치에서 expert $$i$$의 gates를 나열한 것이다.

### 6.4.1. SparseDispatcher.dispatch

- 이제 dispatch에서는 각 expert에 처리해야 할 batches를 할당한다.
- 만약 지금처럼 inp의 크기가 `torch.Size([512, 96, 7, 16])`, `self._batch_index`의 크기가 `torch.Size([1024])`, 그리고 `self._part_sizes`가 `[262, 348, 249, 165]`라고 가정하면:
- `inp[self._batch_index]`에서는 inp 텐서에서 1024개의 샘플을 선택하여, 크기가 `torch.Size([1024, 96, 7, 16])`인 새로운 텐서를 생성한다.
- 그리고 첫 번째 차원(batch 차원, 1024개)을 `self._part_sizes` = `[262, 348, 249, 165]`로 나눈다.
- 결과는 각 expert에게 할당된 batches의 리스트이며, 각 텐서의 크기는:
  - 첫 번째 expert: [262, 96, 7, 16]
  - 두 번째 expert: [348, 96, 7, 16]
  - 세 번째 expert: [249, 96, 7, 16]
  - 네 번째 expert: [165, 96, 7, 16]
- 이걸 리스트로 return한다.

### 6.4.2. SparseDispatcher.combine

- 이제 각각을 해당 expert에 통과시킨다.
- expert는 `TransformerLayer`이다. (Pathformer.py의 __init__참고)
- 그리고 그 결과를 다시 combine한다.
  - 그런데 위에서 combine 함수를 잘 보면 처음에 `.exp()`를 하고 다시 `.log()`를 해주는데,
  - `.exp()`에서 NaN이 나올 수가 있으니 주의하자.
  - (벤치마크 데이터셋에서는 해당사항 없지만 내 프로젝트에서 사용하는 데이터에서는 발생했다.)

![사진1](/assets/img/pytorch/pathformer_code/fig18.png)

- 이제 residual_connection만 적용해주면 끝난다.
- 여기까지가 하나의 `AMS` layer이다.

![사진1](/assets/img/pytorch/pathformer_code/fig16.png)

- 여기서 for 안에 있는 layer가 AMS layer이다.
- 마지막으로 de-normalization을 하면 끝이다.

- 나머지는 위에서 이미 소개한 `3. run.py`와 `4. exp_main.py`가 전부이다.