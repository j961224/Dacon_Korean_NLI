# Dacon_Korean_NLI

대회링크: https://dacon.io/competitions/official/235875/overview/description

## 1. 대회 설명

한국어 문장 관계 분류 경진대회로, klue benchmark에 존재하는 Natural Language Inference Dataset을 활용해 한 쌍의 문장(Premise, Hypothesis)의 관계를 파악합니다.

한국어 문장 관계를 파악하는 알고리즘을 통해, 최대의 Accuracy를 내는 것이 이번 대회의 목적입니다.

## 2. 데이터 - 내부 데이터

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

## 3. 데이터 - 외부 데이터

* KLUE Official Dev Set 2994개 사용

* KorNLI - multinli.train.ko.tsv 데이터 중, 각 label 당 5000개씩 Test Data와 유사한 형태의 데이터 추출 - 15000개 사용

## 4. 모델

### 4-1. self-explainable model

### 4-2. Roberta-large + bilstm

### 4-3. Roberta-large + Concat Last Four Hidden states Model
