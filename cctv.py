# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:07:35 2021
@author: Playdata


분석을 위해 사용되는 문법 및 사용 모듈
Python 기본 문법을 확인
Pandas와 Matplotlib의 기본적 사용법을 확인

분석 내용

국감브리핑 강남3구의 주민들이 
자신들이 거주하는 구의 체감 안전도를 높게 생각한다는 기사를 확인
http://news1.kr/articles/?1911504    

1. 서울시 각 구별 CCTV수를 파악하고, 
2. 인구대비 CCTV 비율을 파악해서 순위 비교
3. 인구대비 CCTV의 평균치를 확인하고 그로부터 CCTV가 과하게 부족한 구를 확인
4. 단순한 그래프 표현에서 한 단계 더 나아가 경향을 확인하고 시각화하는 기초 확인

"""
####################################################
### ------ 서울시 구별 CCTV 현황 분석하기  ----- ###
####################################################

## 작업에 필요한 모듈 import
# 데이터프레임 및 csv, excel을 읽기 위한 모듈
import pandas as pd

# 숫자 관련 모듈
import numpy as np

# 시각화 작업 모듈
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 파이썬이 실행되고 있는 운영체제 관련 모듈
import platform

# --------------------------------------------------#
# =========  분석작업을 위한 1차 전처리 =========== #
# --------------------------------------------------#

### 1. 엑셀파일 읽기 - 서울시 CCTV 현황 : 01. CCTV_in_Seoul.csv
CCTV_seoul =pd.read_csv("./data1/01. CCTV_in_Seoul.csv", encoding='utf-8')

# 컬럼명 확인 : DataFrame.columns / DataFrame.columns[컬럼index]
CCTV_seoul.columns
'''
Out[98]: Index(['기관명', '소계', '2013년도 이전', '2014년', '2015년', '2016년'], dtype='object')
'''


CCTV_seoul.columns[0]
'''
Out[99]: '기관명'
'''

# 컬럼명 변경 : DataFrame.rename(columns={DataFrame.columns[컬럼index] : 변경컬럼명}, inplace=True)
# inplace=True : 변경사항을 바로 해당 데이터프레임에 적용.
CCTV_seoul.rename(columns={CCTV_seoul.columns[0]:"구별"}, inplace=True)

# 변경 여부 확인
CCTV_seoul.columns[0]
'''
Out[101]: '구별'
'''



### 2. 엑셀파일 읽기 - 서울시 인구현황 : 01. population_in_Seoul.xls
pop_Seoul = pd.read_excel("./data1/01. population_in_Seoul.xls")

# 분석에 필요한 부부만 추출하여 읽기 : header=엑셀 행(index)번호, usecols="엑셀컬럼명, 엑셀컬럼명, ..."
pop_Seoul = pd.read_excel("./data1/01. population_in_Seoul.xls", header=2, usecols="B, D, G, J, N")
# =>  B : 구이름, D: 인구수, G:한국인 , J: 외국인, N : 고령자(65세이상)

# 컬럼명 변경 : 자치구:구별 / 계:인구수/ 계1:한국인 / 계2:외국인 / 65세이상고령자:고령자
pop_Seoul.rename(columns={pop_Seoul.columns[0]:'구별', 
                          pop_Seoul.columns[1]:'인구수', 
                          pop_Seoul.columns[2]:'한국인', 
                          pop_Seoul.columns[3]:'외국인', 
                          pop_Seoul.columns[4]:'고령자'
                          }, inplace=True)

pop_Seoul.columns
'''
Out[106]: Index(['구별', '인구수', '한국인', '외국인', '고령자'], dtype='object')
'''





# --------------------------------------------------#
# ==== 분석작업을 위한 2차 전처리 : 데이터파악 ==== #
# --------------------------------------------------#

### 3. CCTV 데이터 파악하기
CCTV_seoul.head()    # 데이터프레임의 상위 5개 데이터 확인 : head(10)
'''
    구별    소계  2013년도 이전  2014년  2015년  2016년
0  강남구  2780       1292    430    584    932
1  강동구   773        379     99    155    377
2  강북구   748        369    120    138    204
3  강서구   884        388    258    184     81
4  관악구  1496        846    260    390    613
'''

CCTV_seoul.tail()    # 데이터프레임의 하위 5개 데이터 확인 : tail(10)
'''
     구별    소계  2013년도 이전  2014년  2015년  2016년
20  용산구  1624       1368    218    112    398
21  은평구  1873       1138    224    278    468
22  종로구  1002        464    314    211    630
23   중구   671        413    190     72    348
24  중랑구   660        509    121    177    109
'''

# '소계' 컬럼을 기준으로 정렬시킨 후, 상위/하위 5개 데이터를 확인
CCTV_seoul.sort_values(by='소계', ascending=True).head()
'''
     구별   소계  2013년도 이전  2014년  2015년  2016년
9   도봉구  485        238    159     42    386
12  마포구  574        314    118    169    379
17  송파구  618        529     21     68    463
24  중랑구  660        509    121    177    109
23   중구  671        413    190     72    348
'''

CCTV_seoul.sort_values(by='소계', ascending=True).tail()  
# 또는
CCTV_seoul.sort_values(by='소계', ascending=False).head()
'''
     구별    소계  2013년도 이전  2014년  2015년  2016년
0   강남구  2780       1292    430    584    932
18  양천구  2034       1843    142     30    467
14  서초구  1930       1406    157    336    398
21  은평구  1873       1138    224    278    468
20  용산구  1624       1368    218    112    398
'''

# '최근증가율' 컬럼 추가 : 2013년도 이전과 그 이후에 대한 CCTV 증가율
# 데이터프레임에 컬럼 추가 : DataFrame['추가컬럼명'] = 추가될 데이터
# 증가율 = (2014년 + 2015년 + 2016년) / 2013년도 이전 * 100
CCTV_seoul['최근증가율'] = (CCTV_seoul['2014년'] + CCTV_seoul['2015년'] + CCTV_seoul['2016년'] ) / CCTV_seoul['2013년도 이전']   * 100

CCTV_seoul.sort_values(by='최근증가율', ascending=False).head()
'''
Out[114]: 
     구별    소계  2013년도 이전  2014년  2015년  2016년       최근증가율
22  종로구  1002        464    314    211    630  248.922414
9   도봉구   485        238    159     42    386  246.638655
12  마포구   574        314    118    169    379  212.101911
8   노원구  1265        542     57    451    516  188.929889
1   강동구   773        379     99    155    377  166.490765
'''



### 4. 서울시 인구 데이터 파악하기
# 결측치 확인 : isnull() / 유일한 값 : unique() 확인 / 행 삭제 : drop([행(index)번호])

# 합계에 해당하는 행삭제
pop_Seoul.drop([0], inplace=True)

# '구별' 컬럼의 유일한 값 확인
pop_Seoul['구별'].unique()
'''
Out[116]: 
array(['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구',
       '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구',
       '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구', nan],
      dtype=object)
'''

# '구별' 컬럼에 결측치 여부 확인
pop_Seoul[pop_Seoul['구별'].isnull()]
'''
Out[117]: 
     구별  인구수  한국인  외국인  고령자
26  NaN  NaN  NaN  NaN  NaN
'''

# 결측치(NaN : Not a Number) 행 삭제
pop_Seoul.drop([26], inplace=True)
pop_Seoul.tail()
'''
Out[119]: 
     구별       인구수       한국인      외국인      고령자
21  관악구  525515.0  507203.0  18312.0  68082.0
22  서초구  450310.0  445994.0   4316.0  51733.0
23  강남구  570500.0  565550.0   4950.0  63167.0
24  송파구  667483.0  660584.0   6899.0  72506.0
25  강동구  453233.0  449019.0   4214.0  54622.0
'''

## '외국인비율'  '고령자비율' 컬럼을 추가
# '외국인비율' = '외국인' / '인구수' * 100
pop_Seoul['외국인비율'] = pop_Seoul['외국인'] / pop_Seoul['인구수'] * 100

# '고령자비율' = '고령자' / '인구수' * 100
pop_Seoul['고령자비율'] = pop_Seoul['고령자'] / pop_Seoul['인구수'] * 100

# 인구수 / 외국인 / 외국인비율 / 고령자 / 고령자비율 을 기준으로 각각 정렬하여 데이터 확인
pop_Seoul.sort_values(by="인구수", ascending=False).head(5)
'''
Out[122]: 
     구별       인구수       한국인      외국인      고령자     외국인비율      고령자비율
24  송파구  667483.0  660584.0   6899.0  72506.0  1.033584  10.862599
16  강서구  603772.0  597248.0   6524.0  72548.0  1.080540  12.015794
23  강남구  570500.0  565550.0   4950.0  63167.0  0.867660  11.072217
11  노원구  569384.0  565565.0   3819.0  71941.0  0.670725  12.634883
21  관악구  525515.0  507203.0  18312.0  68082.0  3.484582  12.955291
'''

pop_Seoul.sort_values(by="외국인", ascending=False).head(5)
pop_Seoul.sort_values(by="외국인비율", ascending=False).head(5)
pop_Seoul.sort_values(by="고령자", ascending=False).head(5)
pop_Seoul.sort_values(by="고령자비율", ascending=False).head(5)
'''
      구별       인구수       한국인      외국인      고령자     외국인비율      고령자비율
9    강북구  330192.0  326686.0   3506.0  54813.0  1.061806  16.600342
1    종로구  162820.0  153589.0   9231.0  25425.0  5.669451  15.615404
2     중구  133240.0  124312.0   8928.0  20764.0  6.700690  15.583909
3    용산구  244203.0  229456.0  14747.0  36231.0  6.038828  14.836427
13  서대문구  327163.0  314982.0  12181.0  48161.0  3.723221  14.720797
'''




# --------------------------------------------------#
# =================== 분석작업 ==================== #
# --------------------------------------------------#

### 5. CCTV 데이터와 인구 데이터 합치고 분석하기
## CCTV_seoul 데이터프레임 + pop_Seoul 데이터프레임 : Pandas.merge(CCTV_seoul, pop_Seoul, on="공통컬럼명")
# on="공통컬럼명" 은 두 개 데이터프레임에 공통컬럼명이 존재할 경
data_result = pd.merge(CCTV_seoul, pop_Seoul, on="구별")

# 만약, 두 개의 데이터프레임에 공통 컬럼명이 없을 경우
# left_on=컬럼명, right_on=컬럼명


## 연도별 강남 3구에 대한 안전부분을 분석하지 않고, 전체에 대한 안전 여부를 분석하기 위해 불필요한 컬럼을 제거
# 컬럼 제거 : del DataFrame['컬럼명']
del data_result['2013년도 이전']
del data_result['2014년']
del data_result['2015년']
del data_result['2016년']


# 분석 작업 및 향후 시각화를 위해 '구별' 컬럼의 데이터를 index 값으로 설정 : DataFrame.set_index('index로 사용될 컬럼명', inplace=True)
data_result.set_index('구별', inplace=True)



### 각 데이터간의 연관성을 위한 상관관계 확인
# 상관관계 : Numpy.corrcoef()

# 고령자비율과 소계간의 상관관계 확인
np.corrcoef(data_result['고령자비율'], data_result['소계'] )
'''
array([[ 1.        , -0.28078554],
       [-0.28078554,  1.        ]])
'''

# 외국인비율과 소계간의 상관관계 확인
np.corrcoef(data_result['외국인비율'], data_result['소계'] )
'''
array([[ 1.        , -0.13607433],
       [-0.13607433,  1.        ]])
'''

# 인구수과 소계간의 상관관계 확인
np.corrcoef(data_result['인구수'], data_result['소계'] )
'''
array([[1.        , 0.30634228],
       [0.30634228, 1.        ]])
'''




### 6. matplotlib를 이용하여 CCTV와 인구현황 그래프로 분석
# 한글깨짐 방지를 위한 설정
plt.rcParams['axes.unicode_minus'] = False

# 운영체제에 맞는 기본 폰트 설정
if platform.system() == 'Darwin':   # OSX 운영체제
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = 'c:/Windows/Fonts/malgun.ttf'    
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print("~~~ sorroy")


# 시각화를 위한 데이터 확인
data_result.head()

# 구별 / 소계 데이터 시각화
plt.figure()
data_result['소계'].plot(kind='barh', grid=True, figsize=(10,10))
plt.show()


# 소계를 기준으로 정렬시킨 후 시각화
data_result['소계'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
plt.show()


# 인구대비 CCTV 비율 시각화
data_result['CCTV비율'] = data_result['소계'] / data_result['인구수'] * 100


data_result['CCTV비율'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
plt.show()


## 인구수와 CCTV비율에 대한 산점도 시각화
plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'], data_result['소계'], s=50)
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()



## 인구수와 CCTV비율에 대한 산점도에 선형회귀선 시각화
fp1 = np.polyfit(data_result['인구수'], data_result['소계'], 1)

f1 = np.poly1d(fp1)

fx = np.linspace(100000, 700000, 100)

plt.figure(figsize=(10,10))
plt.scatter(data_result['인구수'], data_result['소계'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='r')
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()






### 7. 보다 설득력 있는 자료 작업
fp1 = np.polyfit(data_result['인구수'], data_result['소계'], 1)

f1 = np.poly1d(fp1)

fx = np.linspace(100000, 700000, 100)

# 오차 구하기 : 만약 오차가 '-' 값이 나놀수도 있으므로 abs()를 이용하여 절대값으로 변경
data_result['오차'] = np.abs(data_result['소계'] -  f1(data_result['인구수']) )

df_sort = data_result.sort_values(by='오차', ascending=False)
df_sort.head()

# 시각화
plt.figure(figsize=(14, 10))
plt.scatter(df_sort['인구수'], df_sort['소계'], c=data_result['오차'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='r')

for n in range(10):
    plt.text(df_sort['인구수'][n]*1.02, df_sort['소계'][n]*0.98, df_sort.index[n], fontsize=15)

plt.xlabel('인구수')
plt.ylabel('인구당비율')
plt.colorbar()
plt.grid()
plt.show()


#####################################
### 결론 : 서울시에서 다른 구에 비해 강남구, 양천구, 용산구, 서초구, 은평구는 인구대비 CCTV가 많고,
##         그에 비해 강서구, 송파구, 도봉구 등은 인구대비 CCTV 갯수가 부족하다.
## 따라서 강남 3구 전체가 안전하다고 볼수는 없다!!!!
#######################################













