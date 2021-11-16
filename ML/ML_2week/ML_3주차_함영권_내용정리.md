## Estimator 이해 및 fit(), predict()메서드

---

ML모델 학습을 위해 fit()을, 학습된 모델의 예측을 위해 predict() 사용

분류 알고리즘을 구현한 클래스를 **Classfier**로, 

회귀 알고리즘을 구현한 클래스를 **Regressor**로 지칭,

이 두가지를 합쳐서 Estmator라고 부름.

|분류|모듈명|설명|
|---|---|---|
|예제 데이터|sklearn.datasets|사이킷런에 내장되어 예제로 제공하는 데이터셋|
|데이터 분리, 검증 ,파라미터 튜닝|sklearn.model_selection|교차검증을 위한 학습/테스트셋 분리, 그리드 서치로 최적 파라미터 추출 등의 API제공|
|평가|sklearn.metrics|분류,회귀,클러스터링,페어와이즈에 대한 다양한 성능측정|
|ML알고리즘| sklearn.tree|의사 결정 트리 알고리즘 제공|
||sklearn.linear_model|선형회귀 알고리즘 제공|

## 내장된 예제 데이터셋

---

사이킷런에 내장된 예제 데이터셋은 일반적으로 딕셔너리형태로 되어있음.

키는 data, target, target_names, feature_names, DESCR 로 구성되어 있음.

* **data**는 피처의 데이터 세트를 가리킴.  (넘파이배열)

* **target**은 분류 시 레이블 값, 회귀일 때는 숫자 결과값 세트임.  (넘파이배열)

* **target_names** 는 개별 레이블의 이름을 나타냄. (넘파이배열 또는 파이썬 리스트)

* **feature_names**는 피처의 이름을 나타냄.   (넘파이배열 또는 파이썬 리스트)

* **DESCR**은 데이터 세트에 대한 설명과 각 피처의 설명을 나타냄.   (넘파이배열 또는 파이썬 리스트)

## 학습/테스트 데이터셋 분리 - train_test_split()

--- 

``` python
from sklearn.model_selection import train_test_split

...

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, \
    test_size = 0.3, random_state=121)
```
첫번째, 두번째 파라미터는 각각 입력변수와 목표변수를 넣어준다.

test_size = 0.3 의 의미는 3할을 테스트셋으로, 나머지 7할은 학습용으로 사용한다는것,

random_state 파라미터는 그 구분을 어떻게 하느냐에 관여한다. 옵셔널 파라미터이기 때문에 안넣어 주어도 됨.

## 교차 검증

---

위처럼 학습/테스트셋을 분리하더라도 **과적합(Overfitting)** 문제가 발생 할 수 있음.

과적합은 모델이 학습 데이터에만 과도하게 최적화되어 실제 예측을 다른 데이터로 하는 경우 예측 성능이 과도하게 떨어지는 것을 의미.

이러한 문제점을 보완하기 위해 **교차 검증** 사용

* K 폴드 교차 검증

k폴드 교차  검증은 가장 보편적으로 사용되는 교차 검증 기법. 먼저 사용자가 정한 k개의 데이터 폴드 세트를 만들어서 k번 만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행하는 방법.

k = 5 인 경우, 데이터셋을 5등분 한 뒤, 첫번째 반복에서는 처음부터 4개 등분을 트레이닝셋, 마지막 5번쨰 등분을 테스트셋으로, 

두번째 반복에서는 처음부터 3개 등분, 그리고 마지막 5번째 등분을 트레이닝셋으로, 나머지 4번째 등분을 테스트셋으로하여 ... 이런식으로 5번 성능 평가를 함.

5개의 예측 평가를 구했으면 이를 평균하여 k폴드 평가 결과로 반영하면 된다.

``` python
from sklearn.model_selection import Kfold

...

kfold = Kfold(n_split = 5)    #5개의 폴드 세트로 분리하는 kfold객체  생성

...

for train_index, test_index in kfold.split(iris.data):    #kfold객체의 split()을 호출하면 폴드 별 학습용, 검증용 테스트의 row인덱스를 array로 반환.
...
```

## Stratified K 폴드

---

Stratified K 폴드는 불균형한 분포도를 가진 레이블 데이터 집합을 위한 k폴드 방식임.

불균형한 분포도를 가진 레이블 데이터 집합은 특정 레이블 값이 특이하게 많거나 매우 적어서 값의 분포가 한쪽으로 치우치는 것을 말함.

원본 데이터와 유사한 레이블 값의 분포를 학습/테스트 셋에도 유지하는것이 중요!

stratified k 폴드는 이처럼 k폴드가 레이블 데이터 집합이 원본 데이터 집합의 레이블 분포를 학습 및 테스트 세트에 제대로 분배하지 못하는 경우의 문제를 해결해줌.

이를 위해 원본 데이터의 레이블 분포를 먼저 고려한 뒤 이 분포와 동일하게 트레이닝/테스트셋을 분배해줌.

``` python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
...

#label칼럼을 기준으로 분포를 고려, 폴드 별 학습용, 검증용 테스트의 row인덱스를 array로 반환. split()호출시 반드시 레이블 데이터 세트도 추가 입력 필요
for train_index, test_index in skf.split(iris_df, iris_df['label']):
    ...
```

회귀(Regression)에서는 StratifiedKFold를 지원하지않음.

## 교차검증을 간편하게 - cross_val_score()

---

사이킷런의 교차검증을 편리하게 수행할 수 있게 해주는 API = cross_val_score()

``` python
#cross_val_score()의 주요 파라미터와 순서
cross_val_score(estimator, X, y=None, scoring=None, cv=None,...)
```

estimator는 분류알고리즘 클래스 또는 회귀알고리즘 클래스를 의미하고,

X는 입력변수, y는 목표변수, scoring은 예측 성능 평가 지표를, cv는 교차 검증 폴드 수를 의미.

cross_val_score() 수행 후 반환값은 scoring파라미터로 지정된 성능 지표 측정값을 배열 형태로 반환한다.

첫번째 파라미터로 회귀를 넣을 경우 kfold방식으로 분할함.

## GridSearchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에

---

하이퍼 파라미터란, 모델링할 때 사용자가 직접 정해주는 값.

하이퍼 파라미터에 어떤 값을 넣느냐에 따라 모델의 성능에 큰 차이가 있을 수 있기 때문에 모델의 성능을 최대로 높여주는 좋은 하이퍼파리미터를 고르는 것이 중요.

**그리드서치(Grid Search)**는 정해주어야 할 각 하이퍼 파라미터에 넣어볼 후보값을 몇 개씩 정하고, 

그리고 그 후보값의 조합으로 모델을 학습시켰을 때 가장 성능이 좋았던 하이퍼 파라미터 조합을 고르는것. 각 성능은 k겹 교차 검증을 사용해서 계산함.

GridSearchCV는 순차적으로 파라미터를 테스트하므로 수행시간이 오래 걸림.

GridSearchCV 에 들어가는 주요 파라미터로는,

* estimator: classfier, regressor, pipeline이 사용될 수 있음.

* param_grid: 하이퍼 파라미터값들을 지정해둔 key값과 리스트값을 가지는 딕셔너리가 주어짐. 

* scoring: 예측 성능을 측정할 평가 방법을 지정.

* cv: 교차 검증을 위해 분할되는 세트의 개수를 지정.

* refit: 디폴트값은 True이며, True로 생성 시 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습시킴.

``` python
from sklearn.tree import DecisionTreeClassfier
from sklearn.model_selection import GridSearchCV

# 파라미터를 딕셔너리 형태로 지정
X_train, X_test, y_train, y_test = train_test_split(...)
parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
dtree = DecisionTreeClassfier()
...

grid_dtree = GridSearchCV(dtree, param_grid = parameters, cv=3, refit=True)
grid_dtree.fit(X_train, y_train)
```
 학습,평가를 수행하고 cv_result_ 에 그 결과를 저장함.

grid_dtree를 pandas의 DataFrame으로 변환하면 좀 더 쉽게 알아 볼 수 있다.

``` python
#GridSearchCV 결과를 추출해 DataFrame으로 변환 
score_df = pd.DataFrame(grid_dtree.cv_results_)
```

dataframe으로 변환했을때 볼 수 있는 칼럼명들의 의미는,

* params: 수행할 때마다 적용된 개별 하이퍼 파라미터 값을 나타냄.

* rank_test_score: 하이퍼 파라미터별로 성능이 좋은 score순위를 나타냄. 1이 가장 뛰어남.

* mean_test_score: 개별 하이퍼 파라미터별로 CV의 폴딩 테스트 세트에 대해 총 수행한 평가 평균값

GridSearchCV 객체의 fit()을 수행하면 최고 성능을 나타낸 하이퍼 파라미터의 값과 그때의 평가 결과 값이 각각 **best_params_** ,**best_score_** 속성에 기록됨( 즉, cv_result_의 rank_test_score가 1일 때의 값)

## 데이터 인코딩

---
머신러닝을 위한 대표적인 인코딩 방식은 **레이블 인코딩Label encoding**과 **원-핫 인코딩One Hot encoding**이 있음.

* 레이블 인코딩

사이킷런의 레이블인코딩은 LabelEncoder 클래스로 구현.
LabelEncoder를 객체로 생성한 후 fit()과 transform()을 호출해 레이블 인코딩 수행.

``` python
from sklearn.preprocessing import LabelEncoder

items=['TV','냉장고','전자렌지','컴퓨터','선풍기','선풍기','믹서','믹서']

# LabelEncoder를 객체로 생성한 후 , fit( ) 과 transform( ) 으로 label 인코딩 수행. 
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값:',labels)
```

``` python
>>> 인코딩 변환값: [0 1 4 5 3 3 2 2]
```

LabelEncoder 객체의 classes_속성값으로 어떤 문자열 값이 인코딩되었는지 알  수 있음.

``` python
print('인코딩 클래스:',encoder.classes_)
```

``` python
>>> 인코딩 클래스: ['TV' '냉장고' '믹서' '선풍기' '전자렌지' '컴퓨터']
```
LabelEncoder 객체의 inverse_transfrom([인코딩된숫자리스트])로 인코딩된 값을 다시 디코딩 할 수 있음

``` python
print('디코딩 원본 값:',encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))
```

``` python
>>> 디코딩 원본 값: ['전자렌지' '컴퓨터' '믹서' 'TV' '냉장고' '냉장고' '선풍기' '선풍기']
```

* 원-핫 인코딩

원-핫 인코딩은 OneHotEncoder 클래스로 쉽게 변환이 가능.

단, 주의할 점으로는 OneHotEncoder로 변환하기 전에 모든 문자열 값이 숫자형 값으로 변환되어야 한다는 것, 입력 값으로 2차원 데이터가 필요하다는것

``` python 
# 먼저 숫자값으로 변환을 위해 LabelEncoder로 변환합니다. 
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
# 2차원 데이터로 변환합니다. 
labels = labels.reshape(-1,1)
```
판다스의 get_dummies(DataFrame)을 사용하면 문자열카테고리값을 숫자형으로 변환할 필요 없이 바로 변환 가능

## 피처 스케일링과 정규화

---

서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업을 **피처 스케일링**(**feature scaling**)이라고 함. 

머신 러닝 모델에 사용할 입력 변수들의 크기를 조정해서 일정 범위 내에 떨어지도록 바꾸는 것.

피처스케일링을 통해 경사 하강법을 좀 더 빨리 할 수 있게 도와줌.

대표적인 방법으로 **표준화**(**standardization**)과 **정규화**(**normalization**)가 있음. 

표준화는 데이터의 피처 각각이 평균이 0이고 분산이 1인 가우시안 정규 분포를 가진 값으로 변환하는 것을 의미

사이킷런에서 제공하는 대표적인 피처 스케일링 클래스인 MinMaxScaler와 StandardScaler

## MinMaxScaler

---

데이터의 최솟값과 최댓값을 이용, 데이터의 크기를 0과 1사이로 바꾸어줌.(음수 값이 있으면 -1에서 1값으로 변환)

``` python 
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler객체 생성
scaler = MinMaxScaler()
# MinMaxScaler 로 데이터 셋 변환. fit() 과 transform() 호출.  
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform()시 scale 변환된 데이터 셋이 numpy ndarry로 반환되어 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature들의 최소 값')
print(iris_df_scaled.min())
print('\nfeature들의 최대 값')
print(iris_df_scaled.max())
```

## StandardScaler

---

개별 feature를 평균이 0이고, 분산이1인 값으로 변환해줌(표준화)

사이킷런에서 선형회귀, 로지스틱회귀 등은 데이터가 가우시안분포를 가지고 있다고 가정하고 구현됐기 때문에 사전에 표준화를 적용하는 것은 예측 성능 향상에 중요한 요소가 될 수 있다.
