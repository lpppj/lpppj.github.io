---
layout: post
related_posts:
  _
title: 
description: >
  []()
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# 동시에 실험(훈련)하는 방법 (BackgroundRun)

![그림1](/assets/img/pytorch/BackgroundRun/results.png)

- Time series forecasting 실험을 할 때 위 사진처럼 예측할 기간을 96, 192, 336, 720로 설정한다.
- 각각 연달아서 하면 시간이 오래 걸리기 때문에, resource가 허락한다면 **script 파일을 수정**해서 동시에 돌릴 수 있다.

![그림1](/assets/img/pytorch/BackgroundRun/before1.png)

- PatchTST의 ETT data에 대한 script이다.
- 1-7번째 줄에서는 실험 결과가 기록될 log 파일이 저장될 공간을 만들어주고 있다. (필요하다면)
- 8번째 줄에서는 과거 336시점을 보도록 설정하고 있고
- 9번째 줄에서는 PatchTST 모델을 사용할 것임을 설정한다.
- 11-14번째 줄에서 데이터셋을 지정해준다.

![그림1](/assets/img/pytorch/BackgroundRun/before2.png)

- 16번째 줄에서 random seed를 설정하고
- 17번째 줄에서 for loop를 실행한다. 각각은 예측할 기간을 96, 192, 336, 720으로 실험을 한다.
- 19-42번째 줄에서는 hyperparameters를 정해주고, 실험 결과를 기록할 위치를 43번째 줄에서 정한다.
- 하지만 이대로 하면 96시점이 끝나야 192 시점을 시작, ..., 336시점이 끝나야 720시점이 시작된다.
- 96, 192, 336, 720 실험들을 동시에 하기 위해서는 script를 아래와 같이 수정할 수 있다.

![그림1](/assets/img/pytorch/BackgroundRun/after1.png)

- 일단 첫 부분은 동일하다
- terminal에서 명령어를 실행할 때 bash를 사용하기 위해 1번째 줄과 같이 적어주었다.

![그림1](/assets/img/pytorch/BackgroundRun/after2.png)

- 21-22번째 줄에서 각 실험들(예측하고자 하는 미래 시점들)을 몇 번째 gpu에 할당할 것인지를 표시해준다.
- 지금의 경우에는 96과 192는 0번째 gpu, 336과 720은 1번째 gpu를 사용한다는 의미이다.
- 33번째 줄의 `echo`는
  - 실험 결과나 모델이 출력하는 모든 출력들을 `.log` 파일에 기록할 때에도
  - `echo`에 있는 문장은 terminal에 출력된다.
- 35번째 줄의 `CUDA_VISIBLE_DEVICES`에서 현재 사용할 gpu를 설정하는데,
  - 사전에 정해둔 gpu의 순서에 따라 indexing해서 정해준다.
- 59번째 줄에서 마지막에 `&`를 넣어주어야 for loop가 동시에 실행된다.
  - `&`가 없으면 96시점이 끝나야 192 시점을 시작, ..., 336시점이 끝나야 720시점이 시작된다.
- 61번째 줄의 `wait`는
  - 만약 96 시점을 예측하는 실험이 720 시점을 예측하는 실험보다 먼저 끝나더라도 기다리라는 의미이다.
  - 그러므로 96, 192, 336, 720 시점을 예측하는 실험들이 모두 끝나야 for loop가 종료된다.