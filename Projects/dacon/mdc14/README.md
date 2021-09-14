# 신용카드 사용자 연체 예측
* Dacon: https://dacon.io/competitions/official/235713/overview/description

* Project Results: https://www.notion.so/95740d4a98e54722aa79d157d249b9f4

**팀 β** 

참여 : 윤나요 최벽문

작업 환경 : Python 3.8.1

기간 : 2021. 8. 12 ~  2021. 9. 02

[프로젝트 기간](https://www.notion.so/b4206f317dc74b45973d1fccdbc44091)

## 목차

1. 주제 설정 및 목표

2. 데이터
    1. 데이터 설명
    2. 이상치, 결측치
    3. 분석의 방향

1. 데이터 전처리
    1. 이상치와 결측치 수정
    2. 파생변수 생성

2. 모델링
    1. 1차 모델링
    2. 2차 모델링

---

## 1. 주제 설정 및 목표

### 주제 : 신용 카드 불량 이용자를 찾아내는 모델 구성

- 데이터 분석 : 주제에 맞게 데이터를 정제하고, 필요한 변수로 변형하거나 생성한다.
- 모델링 : 주제에 맞게 신용 불량자를 찾을 수 있는 변수를 선택하고, 유의미한 모델을 구성한다.

## 2. 데이터

### a. 데이터 설명

- 출처  : 신용카드 사용자 연체 예측 AI 경진대회

[신용카드 사용자 연체 예측 AI 경진대회](https://dacon.io/competitions/official/235713/overview/description)

[컬럼](https://www.notion.so/ba82632e8ac042ff8ce43149ab5fa54d)

### b. 이상치와 결측치

1. 이상치
    1. family_size 과다

        family size 가 9 이상인 데이터를 제외한다.

        ![Untitled](https://user-images.githubusercontent.com/54128055/132951805-a9236d52-aab6-445a-801a-5fddb59e1f93.png)

        ![Untitled 1](https://user-images.githubusercontent.com/54128055/132951820-65914dea-15d2-419b-8ae9-3d17d05d84df.png)

    2. family_size 가 child_num 보다 적은 데이터를 제외

        성인의 신용 카드 가입 데이터이므로, 아이의 숫자가 전체 가족의 숫자와 같거나 적은 데이터는 데이터 오류로 가정하고, 해당 데이터를 제외한다.

        ![Untitled 2](https://user-images.githubusercontent.com/54128055/132951828-fff0efcf-4418-42b5-a666-e9542260e8fe.png)

    3. DAYS_EMPLOYED 이상치

        DAYS_EMPLOYED 값은 0 부터 음의 방향으로 값을 쌓아가고, 근로기간이 길수록 - 값이 크게 나타난다.

        하지만 값을 rugplot 으로 확인했을 때 아주 큰 양수로 잡혀 있는 값을 확인 할 수 있고, 이 값들은 모두 하나의 값이 입력되어 있다.

        ![Untitled 3](https://user-images.githubusercontent.com/54128055/132951839-5c38f54e-024c-4bfe-98ef-7bcb53303e23.png)

        이 때 양의 정수로 되어 있는 값은 전부 수입이 연금인 케이스로 확인하였다.

        해당 데이터들은 값을 0 으로 조정한다.

        ![Untitled 4](https://user-images.githubusercontent.com/54128055/132951850-c8353106-94d3-4679-a207-80244035667b.png)

2. 결측치
    1. occyp_type

        주어진 데이터에서 결측치는 occyp_type 에만 있다.

        ![Untitled 5](https://user-images.githubusercontent.com/54128055/132951855-32c42a6e-018d-46ef-b809-19cc3b65ee0c.png)

        ![Untitled 6](https://user-images.githubusercontent.com/54128055/132951861-c42fde1d-6246-40eb-ab14-1f714f9026e5.png)

        - 이 때 결측치의 절반인 Pensioner 는 연금생활자로  'Unemployed'로 구분한다.
        - 나머지 항목은 미기입으로 판단되지만, 해당 데이터의 숫자가 전체 데이터의 15% 정도로 적지 않으므로, 제거하지 않고, 'Unknown' 으로 분류한다.

### c. 분석의 방향성 설정

1. 변수 heatmap 확인

    - 범주형 변수 별로 타겟 변수인 'credit' 의 비율을 살펴보았을 때, 표본이 극소수인 일부 변수를 제외하고 대부분의 변수 값에서 타겟 변수의 비율이 비슷하다.

        어느 컬럼을 살표보아도 타겟 변수 [0, 1, 2] 와 [0.12, 0.22, 0.66] 수준의 비슷한 비율을 나타내기 때문에 기존 변수만을 살펴보았을 때는 타겟 변수와의 상관성이 있는 항목을 추려내기가 어렵다.

    ![Untitled 7](https://user-images.githubusercontent.com/54128055/132951866-1a13379d-215d-438e-a675-ecff0c00e1b9.png)

    - 수치형 변수의 경우도 타겟 변수인 신용도를 기준으로 파악하였을 때는, 고루 분포되어 있어 눈에 띄는 연관성을 파악하기 힘들었다.

    ![Untitled 8](https://user-images.githubusercontent.com/54128055/132951871-7d017081-3a53-46b1-9719-e17747f6b7f6.png)

     

2. 동일한 데이터가 동일한 결과를 보이지 않음

    - 'index'와 'begin_month' 컬럼만 상이하고, 모든 게 동일한 데이터의 집합에서도 데이터의 결과라고 할 수 있는 타겟 변수의 값이 상이함을 볼 수 있다.

        이때 index 는 고유번호로 데이터의 결과 값에 아무런 영향을 주지 않는다.

        ![Untitled 9](https://user-images.githubusercontent.com/54128055/132951877-2733cf9e-783e-40c4-b769-909dbce5648a.png)

    - 'begin_month' 컬럼의 경우 개별 컬럼으로서는 결과값에 뚜렷한 상관성을 보인다고 할 수 없다.

        ![Untitled 10](https://user-images.githubusercontent.com/54128055/132951884-5a177704-059d-4e11-9170-b579df67a43d.png)

    - 고유변수와 무작위인 데이터를 제외하면 모두 같은 입력값을 가질 때 같은 결과를 나타내야 하지만, 실제로는 데이터가 그렇지 않음이 확인되었다.

        데이터의 입력값과 출력값이 일치하지 않은 문제는 추후 모델을 학습하는 과정에서 문제가 생길 것으로 예상 되었다.

**분석의 방향성 설정**

데이터를 살펴보았을 때, 이상치나 결측치는 비교적 명확하게 해결이 가능했지만, 주어진 데이터에서 신용등급 분류를 위한 척도를 찾기가 어려웠고, 동일하다고 여겨지는 데이터에서 다른 결과값이 나오는 점이 문제가 되었다. 이를 해결하기 위해 다음의 방향으로 분석을 진행하였다.

1. 기존 변수 외에 타겟 변수를 좀 더 다양한 기준으로 볼 수 있는 파생 변수 생성
2. 동일하다고 여겨지는 데이터의 취급
    1. 이 데이터들을 다룰 방법의 정의
    2. 동일 데이터의 다른 결과값을 수정할 수 있는 지에 대한 논의
    3. 수정한다면 어떤 방향으로 수정할 지에 대한 논의

---

## 3. 데이터 전처리

1. 이상치와 결측치 처리

    ```python
    # family_size 이상치 제거
    train = train[train['family_size'] <= 8]

    # family_size 와 child_num 관계 이상치 제거
    train = train[(train['family_size'] - train['child_num']) > 0][['child_num','family_size']]

    # pensioner 근로일수 0 처리
    train['DAYS_EMPLOYED'] = train['DAYS_EMPLOYED'].apply(lambda x : 0 if x > 0 else x)

    # occyp_type 결측치 채우기
    cond = train['income_type'] == 'Pensioner'
    train['occyp_type'] = train['occyp_type'].fillna(cond.map({True:'Unemployed', False: 'Unknown'}))
    ```

2. 파생변수 생성
    1. age, work

        **age** : 'DAYS_BIRTH' 데이터를 이용 각 이용자의 나이로 변환(만 나이)

        **work** : 'DAYS_EMPLOYED' 데이터를 이용, 각 이용자의 근속년수를 연차로 변환

        ```python
        # 음수 데이터 양수화
        df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x : np.abs(x))
        df['begin_month'] = df['begin_month'].apply(lambda x : np.abs(x))

        # 나이(일수)
        df['age'] = df['DAYS_BIRTH'].apply(lambda x : x//365)

        # 근속연수 치환
        df['work'] = df['DAYS_EMPLOYED'].apply(lambda x : (x//365 +1) if x > 0 else 0)
        ```

        ![Untitled 11](https://user-images.githubusercontent.com/54128055/132951896-080f16fc-6e2a-4bbd-99f4-2703b074bebe.png)

    2. 중복 데이터 처리

        'begin_month' 를 제외하고 동일한 데이터를 가진 그룹을 동일한 데이터를 가진 것에서 한 걸음 더 나아가 동일인으로 가정하고, 동일인의 각 갯수를 중복 계정 변수로 활용한다.

        ![Untitled](%E1%84%89%E1%85%B5%E1%86%AB%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%8F%E1%85%A1%E1%84%83%E1%85%B3%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%8C%E1%85%A1%20%E1%84%8B%E1%85%A7%E1%86%AB%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%2000efa35f8a984970936ffa400c705c24/Untitled%209.png)

        **account** : 동일한 정보를 가진 그룹의 숫자를 각 행에 account 로 추가

        **expired** : 'begin_month' 이 60 개월이 되어 사용 기간이 만료가 된 계정 수를 구하고 마찬가지로 동일인을 기준으로 업데이트

        **account_live** : 동일인 기준으로 사용 중인 카드의 숫자를 추가

        ```python
        # 중복인 처리

        # 만료된 계정 확인
        df['expired'] = df['begin_month'].apply(lambda x : 1 if x == -60 else 0)
        df['expired'] = df.groupby(dople_cols)['expired'].transform(sum)

        # 중복 계정
        df['account'] = df.groupby(dople_cols)['income_total'].transform(len)

        # 만료 되지 않은 계정
        df['account_live'] = df['account'] - df['expired']
        ```

        ![Untitled 12](https://user-images.githubusercontent.com/54128055/132951904-441406c8-4b2f-4030-ba53-ddaa5c7f98d0.png)

    3. income

        신용도에 영향이 클 것이라고 생각되는 수입을 여러 기준으로 분리해서 고려

        **income_log** = 'income_total' 을 log 처리

        **income_avail** = 'income_total' 을 현재 사용 중인 'account' 로 나누어, 사용하는 카드 별로 분배 가능한 수입액을 계산

        **income_avail2** = 'income_total' 을 'child_num' 으로 나누어 부양가족 별로 사용 가능한 수입액을 계산

        ```python
        # 수입 컬럼 / 로그
        df['income_log'] = df['income_total'].apply(lambda x : np.log(x))
        # 수입 / 사용 계정수
        df['income_avail'] = np.log(df['income_total'] /df['account_live'])
        # 수입 / 부양 가족수
        df['income_avail2'] = np.log(df['income_total'] / (df['child_num']+1))
        ```

        ![Untitled 13](https://user-images.githubusercontent.com/54128055/132951911-072a6287-46c9-462a-ad1d-e9d30ba24e1b.png)

    4. period

        신용카드를 많이 사용하는 정도를 총합과 평균으로 확인

        ```python
        # begin_month 를 동일인 기준으로 총합 / 평균
        # 총합
        df['period_total'] = df.groupby(dople_cols)['begin_month'].transform(sum)
        # 평균
        df['period_avg'] = df['period_total'] / df['account']
        ```

        ![Untitled 14](https://user-images.githubusercontent.com/54128055/132951926-ac7ae9f4-0216-489e-a992-585ba3962bfa.png)

---

## 4. 모델링

[모델링에 사용한 변수](https://www.notion.so/9e571351f0984789aed4e651331cebdf)

## 모델 선택 : Light GBM

- kaggle 분류 문제에서 높은 성적을 거두는 GBM 계열로 선택
- Tree base 모델들은 자체적으로 제공하는 Feature Importance 기능을 통해 변수의 중요도를 확인하며 feature selection에 큰 도움이 될 수 있다고 판단
- category type 의 컬럼이 많아 one-hot encoding 이 필요한 XGBoost 는 너무 많은 컬럼이 생성되기 때문에 효율성을 생각하여 제외
- Laptop Local 환경으로 해당 과제를 진행하였기에, 다양한 시도를 할 수 있도록 성능이 안정적이고 모델 적합 속도가 빠른 Light GBM 모델을 최종적으로 선택하였다.

[참고) GBM Models](https://www.notion.so/714101b9737f40f0b08262dedc7a3bdd)

## Model_Model Training

### K-Fold Cross Validation → 자체적인 평가 데이터 이용

모델링 과정에서 과적합을 방지하기 위해 제공된 'train' 데이터 셋을 train 과 validation 으로 나누어서 모델을 교차 검증
데이터 셋을 k=n 개로 나누어 n-1 개로 모델을 학습하고 1개로 모델을 검증하는 K-Fold 교차 검증 방식으로 모델을 평가
특정 class의 빈도가 높게 쪼개지는 것을 방지하고 과적합을 피하기 위해 stratified k-fold로 진행

- K = 5 일 때는 class 0의 개수가 상대적으로 적게 잡히기 때문에 1개의 fold가 전체 데이터를 대표하기에는 부적절하다고 판단하여 K를 15로 선정 (k 가 15일 때, 각 클래스 간 비율은 같으나 표본의 개수가 더 많음)
- 해당 데이터 셋은 대회용 데이터 셋이기 때문에 test 데이터의 예측값을 제출하여 검증해 볼 수 있으므로 적절한 k를 제출해가며 찾는 방법 또한 적절하다 판단

```python
n_splits = k
skf = StratifiedKFold(n_splits, shuffle= True, random_state= 42)
```

k = 5 일 때,

<img width="302" alt="Screen_Shot_2021-09-01_at_12 29 17_PM" src="https://user-images.githubusercontent.com/54128055/132951980-494df899-b460-4cc3-8a23-2cf0ec471314.png">

k = 15 일 때,

<img width="302" alt="Screen_Shot_2021-09-01_at_12 31 01_PM" src="https://user-images.githubusercontent.com/54128055/132951987-0c32e773-af4a-4091-8869-5ae79164cb64.png">

### Out-of-Fold Ensemble

KFold 교차 검증에서의 각 Fold에 대한 예측 값을 앙상블 하며 동시에 모델 검증을 할 수 있기 때문에 OOF로 예측 값을 산출

 각 fold마다 예측된 확률들을 합하고 이를 위에서 설정한 k(15) 개로 나누어 각 3개의 class에 속하게 될 최종 확률을 도출 

- 평가 지표는 Log Loss로 모델을 평가 (Dacon에 제출 시 Log Loss로 제출해야 되는 대회 규정)
- 모델 학습 시 validation data의 [log loss](https://www.notion.so/Log-Loss-02c2eaf4cb814b129d02fc84b477b71a)를 minimize

[Result: Class 0, 1, 2 (낮을 수록 고신용도) ](https://www.notion.so/ec614b20b3c6420894d606c8bb668ba8)

### Account Feature

- 해당 변수는 중복 row의 개수로 한 사람이 보유하고 있는 신용 카드의 개수로, 데이터에서 중복인이 있음을 가정한다면 가장 중요한 파생변수라고 볼 수 있다.
- 하지만 실제로 모델 학습 시 validation data에 대한 log loss 는 줄어드는 양상을 보인 반면, 대회 'test' 데이터 셋에 대해서는 loss 가 증가하여 모델이 과적합 되는 결과를 보였고, 대회 기준으로 모델을 구성한다고 하면 오히려 방해가 되는 양상을 보였다.

    <img width="663" alt="Screen_Shot_2021-09-01_at_1 04 43_PM" src="https://user-images.githubusercontent.com/54128055/132951994-245f6ec2-f1b1-46aa-9181-fcaeeed5a99e.png">

- Account 변수를 사용한 모델로 Test 데이터를 예측하였을 때, 오히려 성능이 떨어지는 모습을 보인다.

    ![Untitled 15](https://user-images.githubusercontent.com/54128055/132952000-f44d03ee-0406-49a9-b860-43e69f8b468b.png)

## 모델링 결과

1. **모델링 점수가 좀처럼 향상되지 않았다.**
    - 모델링의 결과는 **log loss 0.6963**으로 대회 리더보드와 비교하였을 때, 우수한 결과가 아니었다. 하지만 리더보드 1위의 스코어도 0.6581 정도로 해당 데이터를 가지고 구성한 모델의 성능이 썩 뛰어나게 나오지 않음을 확인하였다.
    - 스코어를 확인했을 때 해당 모델은 신용 불량자를 예측한다는 목적에 부합하기 어렵다고 판단할 수 있는데, 이 원인을 타겟 변수의 오류에서  찾아보기로 하였다.

        <img width="1374" alt="Screen_Shot_2021-09-01_at_4 54 05_PM" src="https://user-images.githubusercontent.com/54128055/132952005-e72b5abe-1e5a-4c18-b330-90f33474b532.png">

2. **타겟 변수 'credit' 은 동일한 입력에 동일한 결과값을 갖지 않는다.**
    - 이런 데이터의 문제는 학습 시 모델이 일관성 있는 분류를 하지 못 하도록 방해가 된다.

3. **데이터 셋의 타겟 변수 'credit' 은 조정되어야 한다.**
    - 모델 학습을 저해하는 요소가 결과값 'credit' 의 불일치 때문이라면, 이 변수 값을 조정해줘야 할 필요가 있다.

        물론 모델의 타겟 변수를 조정하는 건 대회를 참가하는 입장에서는 적용할 수 없는 방법일 수 있지만, 신용 불량자를 구분해낸다는 목적을 감안한다면 오히려 더 적합한 시도라고 생각했다. 

        타겟 변수를 조정하는 방법은 2가지로 추려졌다.

    1. 그룹의 신용도 중 가장 낮은 신용도로 조정한다. 

        이는 신용 카드를 여러 개 사용하는 사람일수록 자산 관리가 방만하고, 장기적으로 이미 신용도가 하락한 단계까지 다른 카드의 신용도가 하락할 것이라는 가정에서 나온 방법이다.

    2. 그룹의 신용도 빈도가 가장 높은 값으로 다른 값을 대치한다.

        이는 카드 당 신용도가 각각의 독립적인 평가의 결과로 나타난 것이고, 가장 빈도가 높은 신용도로 수렴할 것이라는 가정에서 나온 방법이다.

    - 최종적으로 타겟 변수의 수정은 b 안(최빈값)을 선택하여 비교적 온건한 방향으로 조정하였다.

---

## 모델링 : 타겟 변수 'Credit_Mode'

- 중복이 존재하는 데이터를 그룹화하였을 때 타겟 변수를 최빈값으로 조정

    ```python
    print("중복의 개수가 10가 넘는 instance: {0}".format(len(data[data["account"]>10])))
    print("중복의 개수가 5가 넘는 instance: {0}".format(len(data[data["account"]>5])))
    print("전체 데이터셋 중에 credit과 credit_mode가 일치하지 않는 경우: {0}".format(sum(data["credit"]!= data["credit_mode"])))

    중복의 개수가 10가 넘는 instance: 2088
    중복의 개수가 5가 넘는 instance: 10198
    전체 데이터셋 중에 credit과 credit_mode가 일치하지 않는 경우: 4897
    ```

- target variable인 "credit"을 → "credit_mode"로 학습 시도

    기존의 데이터를 나누는 KFold에서 target_mode만 추가하여, 모델 학습 시 타겟 변수를 y_train_mode으로 변경

    <img width="1171" alt="Screen_Shot_2021-09-01_at_6 34 29_PM" src="https://user-images.githubusercontent.com/54128055/132952011-4218395e-36af-4be6-a8aa-66367822eff6.png">

- valid [logloss](https://www.notion.so/Log-Loss-02c2eaf4cb814b129d02fc84b477b71a) 값을 보면, 기존의 target으로 학습하는 것보다 target_mode로 학습한 multi_log loss가 작다는 것을 알 수 있다.

    <img width="679" alt="Screen_Shot_2021-09-01_at_6 02 16_PM" src="https://user-images.githubusercontent.com/54128055/132952015-bfadd96f-0dcb-465f-9dcf-414bbc1c507e.png">

### 결론

1. **타겟 변수를 조정한 후 모델의 예측력이 훨씬 향상되었다.**

    타깃 변수를 조정해 준 뒤 다시 모델링 한 점수는 **log loss 0.2162**로 기존 모델의 점수 **0.6963**  대비 훨씬 나아진 결과를 보인다. 

    이는 동일한 입력에 대해 동일한 결과값을 가지고 학습한 모델이 보다 일관성있고, 명확한 예측을 한 것이지만,

    아쉽게도 대회 Test 데이터를 기준으로는 확인이 불가능한 방법이다.

2. **목표로 했던 모델 구성이 원활하지 않을 때, 길을 돌아갈 수도 있다.**

    해당 분석에서 타겟 변수를 조정함으로써 결론의 신뢰성이 다소 무너질 수 있는 상황이라고 생각한다.

    하지만 실제로 해당 데이터를 받아 분석을 해야 한다면, 제공된 데이터 만으로는 명확한 결론을 낼 수 없는 상황이었고, 추가로 데이터를 찾아봤을 것이다.

    하지만 추가로 데이터 수집이 불가능한 상황에서 분석의 목적인 '신용 불량자를 예측한다' 에 맞는 보다 합당한 방법으로 타겟 변수를 조정하는 시도는 충분히 합리적인 방향이었다고 생각하며, 이상 분석을 마친다.
