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

# (Corrformer, NMI 2023) Code Review 0 - Code Structure

![사진1](/assets/img/pytorch/corrformer0/corrformer01.png)
![사진2](/assets/img/pytorch/corrformer0/corrformer02.png)

- [Corrformer github](https://github.com/thuml/Corrformer)에서 `Corrformer.sh`를 확인하면 되겠다.

### Corrformer.sh

![사진3](/assets/img/pytorch/corrformer0/corrformer03.png)

- `run.py`를 보도록 한다.

### run.py

![사진4](/assets/img/pytorch/corrformer0/corrformer04.png)

- `exp.train`, `exp.test`, `exp.predict`를 실행하는데 `exp`는 `Exp(args)`이고, `Exp`는 `Exp_main`이다.

![사진5](/assets/img/pytorch/corrformer0/corrformer05.png)

- `Exp_main`은 `exp` 폴더 안에 있는 `exp_main.py`에 있는 함수 `Exp_Main`이 되겠다.

### exp_main.py

![사진6](/assets/img/pytorch/corrformer0/corrformer06.png)

- `Exp_Main`이라는 함수에서 `_build_model`에서 `Corrformer`를 불러온다.
- (수정) 그런 줄 알았는데 `Exp_Main`의 `train`에서 `_build_model`가 실행되지 않는다 !
- 그 말은 `_build_model`에서가 아니라 `Corrformer.sh`에서 받은 parameter `Corrformer`를 그대로 받음을 의미한다.
- 결과적으로는 `self.model`에 Corrformer가 들어간다는 건 변함이 없다.
- 아무튼 앞서 `run.py`에서는 `Exp_Main.train`을 실행했기 때문에 다음 포스팅부터는 `train` 함수부터 보도록 한다.