# Dacon_Korean_NLI

[대회링크](https://dacon.io/competitions/official/235875/overview/description)

## 1. 대회 설명

한국어 문장 관계 분류 경진대회로, klue benchmark에 존재하는 Natural Language Inference Dataset을 활용해 한 쌍의 문장(Premise, Hypothesis)의 관계를 파악합니다.

한국어 문장 관계를 파악하는 알고리즘을 통해, 최대의 Accuracy를 내는 것이 이번 대회의 목적입니다.

## 2. 개발 환경

GPU: Colab Pro P100 2개

## 3. 사용 라이브러리

```
torch==1.6.0
pandas==1.1.5
scikit-learn==0.24.1
transformers==4.10.0
wandb==0.12.1
dataset==1.18.3
```

## 4. 파일 구조

```
Dacon_Korean_NLI/
├── utils/
│   ├── collate_functions.py
│   ├── preprocessor.py
│   └── trainer.py
├── models/
│   ├── ConcatLastHiddenModel.py
│   ├── Explainable_Model.py
│   └── transformers_with_bilstm.py
├── train.py
├── train_mlm.py
├── load_data.py
├── papago_backtranslation.ipynb
└── inference.py
```


## 5. 데이터 - 내부 데이터

train data 24998개, test data 1666개로 구성!

train data의 columns: 'label','premise','hypothesis','index'

* train data label 분포

~~~
entailment       8561
contradiction    8489
neutral          7948
~~~

* train data의 premise와 hypothesis 길이 분포

premise 길이는 전반적으로 고루 분포되어 있음을 알 수 있습니다.

![length길이](https://user-images.githubusercontent.com/59636424/156333121-94da847c-44f9-40b0-8e61-09973aeecf12.PNG)

## 6. 데이터 - 외부 데이터

* [KLUE Official Dev Set](https://klue-benchmark.com/tasks/68/data/download) 2994개 사용

* [Kakaobrain KorNLI Dataset](https://github.com/kakaobrain/KorNLUDatasets) - multinli.train.ko.tsv 데이터 중, 각 label 당 5000개씩 Test Data와 유사한 형태의 데이터 추출 - 15000개 사용

## 7. 모델

### 7-1. klue/roberta-large + [self-explainable model](https://arxiv.org/pdf/2012.01786.pdf)

![논문 사진](https://user-images.githubusercontent.com/59636424/156876436-de16cd67-556e-436b-8c56-b148d66c1955.PNG)

self-explainable model은 text span을 통해 self-explain한 모델을 구상

* SIC layer: 각 span들에 대해 hidden state vector를 얻는다.
* interpretation layer: text span에 대한 정보를 집계하여 weight sum으로 weighted combination을 만든다.
* output layer: softmax를 통해 최종 확률값을 출력한다.

이번 대회에서는 klue/roberta-large 모델을 Intermediate layer로써 사용. 

### 7-2. klue/roberta-large + bilstm

klue/roberta-large + bilstm 2개 layer를 통해 2개의 hidden states를 concat하여 예측한 모델

### 7-3. klue/roberta-large + [Concat Last Four Hidden states Model](https://www.kaggle.com/rhtsingh/utilizing-transformer-representations-efficiently)

klue/roberta-large의 output으로 hidden states를 받아 그 중, 마지막 4개의 hidden states를 concat하여 예측한 모델

## 8. 모델 훈련 방식 및 전략

### 8-1. Stratified K-Fold 활용

전체적으로 5-Fold까지 사용

### 8-2. mlm pretrain 방식

Masked Language Model pretrain 방식을 train 데이터를 활용해 klue/roberta-large에 pretrain 적용

* 전체 token 중, 15%를 [MASK] token으로 변환 -> 80%는 [MASK]토큰으로 변환하고 10%는 랜덤 토큰으로 변경하며, 10%는 변환

## 9. 결과

### 9-1. Hyperparameter

* Learning Rate: 2e-5
* batch size: 16 * 2(accumulation step) = 32
* warmup steps: 500
* optimizers: adamw
* scheduler: get_linear_schedule_with_warmup, noam scheduler

### 9-2. Final Results

Model1: **(기존 train data + KLUE Official Dev Set)** klue/roberta-large & Explainable Model +  klue/roberta-large & bilstm Model + klue/roberta-large + **(기존 train data + KLUE Official Dev Set & mlm pretrained)** klue/roberta-large + **(기존 train data + multinli 15000개 data)** klue/roberta-large + **(기존 train data + multinli 15000개 data & mlm pretrained)** klue/roberta-large & Explainable Model

Model2(단일 모델): **(기존 train data + KLUE Official Dev Set)** klue/roberta-large & Explainable Model

|            | Public Accuracy | Private Accuracy |
|:----------:|:------:|:------:|
| Model1 | 0.894 | **0.88775915** |
| Model2 | 0.891 | 0.8871548619 |

Final submission: Model1 제출

Public & Private Rank: 18/468 | 19/468

## 10. 참고자료

* [self-explainable Model paper](https://arxiv.org/pdf/2012.01786.pdf)
* [Last Four Hidden States Concat](https://www.kaggle.com/rhtsingh/utilizing-transformer-representations-efficiently)
