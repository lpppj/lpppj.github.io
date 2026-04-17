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

# (Corrformer, NMI 2023) Code Review 1 - Overall Framework

![사진11](/assets/img/pytorch/corrformer0/corrformer011.png)

- `_get_data` 함수로 데이터를 불러오는데 예를 들어 train_data의 경로는 `Corrformer/dataset/global_temp/temp_global_hourly_train.np`이고 shape은 `(12280, 3850, 1)`이다.
  - 12280은 timestep의 개수, 3850은 sensor의 개수이다.
- `train_steps = len(train_loader)`는 `len(self.data_x)` - `self.seq_len` - `self.pred_len` + `1` = `12280 - 48 - 24 + 1` = `12209`가 된다.
  - 12209번째부터 마지막 12280째까지 (48+24=72)개가 마지막 배치에 들어가고 12210째부터는 72개를 만들 수 없기 때문
- `_select_optimizer`로 MSE, `_select_criterion`로 Adam을 사용한다.

  - 참고로 `_get_data`는 `data_factory.py`에 있는 `data_provider` 함수를 import한 것이다.

![사진8](/assets/img/pytorch/corrformer0/corrformer08.png)

- `data_factory.py` - `data_provider` 함수의 첫 줄에서 `arg.data`에는 `Corrformer.sh`에서 정한 Global_Temp 또는 Global_Wind가 들어간다.
- 그러면 `data_loader.py`에서 실제로 데이터를 불러오는 코드가 실행된다. 
- `data_loader.py` 코드를 보지는 않겠지만 numpy version 호환성으로 인해 `astype(np.float)`로 적힌 두 곳을 `astype(np.float64)`로 바꿔주었다. (안바꾸면 버전 호환성으로 인한 에러 발생)
- 이제 매 epoch에서 어떻게 실행되는지 알아보자.

![사진7](/assets/img/pytorch/corrformer0/corrformer07.png)

- 모델을 train mode로 설정하고 `train_loader`에서 데이터를 배치 단위로 불러온다.

~~~python
Corrformer.sh에서 --seq_len 48 \ --label_len 24 \ --pred_len 24 이므로

i : 0
shape of batch_x : torch.Size([1, 48, 3850])
shape of batch_y : torch.Size([1, 48, 3850])
shape of batch_x_mark : torch.Size([1, 48, 4])
shape of batch_y_mark : torch.Size([1, 48, 4])

48개를 보고 24개를 예측하는데 y의 길이도 48인 이유
--> x는 [index 부터 index + seq_len 까지] 이렇게 48개이고 ex. [1, 48]
--> y는 [index + seq_len - label_len 부터 index + label_len + pred_len 까지] 이렇게 48개 ex. [25, 72]
--> [1, 24]는 예측에 사용되는 부분이지 맞춰야 할 부분은 아니기 때문에, x만 있으면 되고 y에는 없어도 되는 것이고
--> [25, 48]은 [1, 24]로 예측도 해야하고 [49, 72]를 예측할 때 사용도 해야 하니 x와 y가 모두 있어야 하고
--> [49, 72]는 맞추기만 하면 되지 다른 예측에는 안쓰이기 때문에 y만 있으면 되는 것
~~~

- decoder의 input `dec_inp`는 `label_len` 만큼의 `batch_y`와 그 뒤에 `pred_len`만큼의 zero를 붙여서 만든다.

~~~python
>>> dec_inp # torch.Size([1, 48, 3850])
tensor([[[ 42.,  67.,  87.,  ..., 300., 310., 320.],
        [ 37.,  67.,  88.,  ..., 310., 320., 340.],
        [ 31.,  70.,  91.,  ..., 320., 330., 350.],
        ...,
        [  0.,   0.,   0.,  ...,   0.,   0.,   0.],
        [  0.,   0.,   0.,  ...,   0.,   0.,   0.],
        [  0.,   0.,   0.,  ...,   0.,   0.,   0.]]])
~~~

- `output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)`에서 `Model` 클래스의 `forward`도 호출
  - `Corrformer.py`에서 `class Model(nn.Module):`로 Model이라는 class를 정의할 때 `nn.Module`을 상속받았는데, `nn.Module`에 오버라이딩 된 `__call__` 메소드 덕분에 `class Model`의 `forward` 메소드도 같이 호출되기 때문이다.
  - 즉 `batch_x`가 `Model` 클래스 `forward`의 `x_enc`로 들어가게 된다.
  - 이 때 `nn.Module`의 `__call__`에 의해 `forward`라는 메소드만 특별하게 처리되는 것이고, 만약 `Corrformer.py`에 `forward` 말고 다른 함수 `custom_function`도 있었다면 `self.model.custom_function`이렇게 해줘야 한다.

- 지금까지는 모델에 대해서 알아본 건 아니고 기본적인 코드의 구조를 이해했다. (아래 사진)
- `exp_main` $$\to$$ `Corrformer.py` 부분은 다음 포스팅에서 다룰 것이다.

![사진10](/assets/img/pytorch/corrformer0/corrformer010.jpeg)

![사진9](/assets/img/pytorch/corrformer0/corrformer09.png)

- 다음 포스팅부터는 `outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)`에서 무슨 일이 일어나는지 알아보기 위해 `Exp_Main`에서 `arg.model`로 사용하게 되는 `Corrformer.py`에 있는 `Model` 클래스를 살펴보면서 모델을 이해해보도록 한다.