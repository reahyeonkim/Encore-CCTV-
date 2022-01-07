# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:22:56 2021

@author: Playdata

pandas_basic.py : 판다스 기본 사용법
"""

# 사용할 모듈 import
from pandas import Series, DataFrame

# Series, DataFrame 는 각각 클래스로 선언되어 있슴..

#### 1. Series 사용 방법
kakao = Series([92600, 92400, 92100, 94300, 92300])

'''
Series 객체 생성 시, 별도의 index값을 설정 하지 않으면 
기본적으로 0부터 시작하는 정수로 index가 생성된다.
'''

# 2. index를 별도로 설장하여 Series 객체 생성

# Series의 인덱스로 사용할 리스트 생성
idx = ['2021-12-01', '2021-12-02', '2021-12-03', '2021-12-04', '2021-12-05']

# Series 객체 생성 시, 인덱스를 별도로 설정할 경우 : index=리스트 
# 단, 데이터의 갯수와 인덱스의 갯수가 동일!!!
# Series의 데이터도 리스트로 생성하여 전달 가능

# Series의 데이터로 사용될 리스트 선언
data = [92600, 92400, 92100, 94300, 92300]

# idx와 data를 이용한 Series 객체 생성
kakao2 = Series(data, index=idx)

# Series의 데이터 선택 방법 : index를 이용 (예: kakao[0] / kakao2['2021-12-01'])
print(kakao[2])
print(kakao2['2021-12-02'])

# Series의 전체 데이터 추출하는 방법 : index와 values를 이용
for date in kakao2.index:   # index 값
    print(date)

for price in kakao2.values: # 데이터
    print(price)

### 3. Series 끼리의 연산
mine = Series([10, 20, 30], index=['naver', 'kt', 'sk'])
friend = Series([10, 30, 20], index=['kt', 'naver', 'sk'])

merge = mine + friend




#### DataFrame 생성 방법 : 주로 딕셔너리를 사용
# 딕셔너리를 통해 각 컬럼에 대한 데이터를 저장한 후, 
# DataFrame의 생성자에게 전달.

# DataFrame 객체 생성을 위한 딕셔너리 생성
raw_data = {'col0':[1,2,3,4], 'col1':[10,20,30,40], 'col2':[100, 200, 300, 400]}
dataframe_data = DataFrame(raw_data)
'''
딕셔너리를 이용하여 데이터프레임 객체를 생성하면
딕셔너리의 key가 데이터프레임의 컬럼명으로 자동 인덱싱 되고,
딕셔너리의 value에 해당하는 row에는 리스트처럼 0부터 시작하는 정수로 index가 인덱싱된다
'''

# DataFrame의 데이터 추출 방법 1 : 컬럼명을 이용
print(dataframe_data['col0'])

# DataFrame의 컬럼은 Series 형태이다.
type(dataframe_data['col0'])   # pandas.core.series.Series

# DataFrame의 컬럼명을 별도로 설정 : columns = 리스트
# 딕셔너리 데이터 생성
daeshin = {'open': [11650, 11100, 11200, 11100, 11000],
           'high': [12100, 11800, 11200, 11100, 11150],
           'low' : [11600, 11050, 10900, 10950, 10900],
           'close': [11900, 11600, 11000, 11100, 11050]}

daesin_day = DataFrame(daeshin)

# 컬럼명 순서 변경 : columns =[딕셔너리의 key명들]
daesin_day2 = DataFrame(daeshin, columns=['open', 'low', 'close', 'high'])

# DataFrame의 index 설정 : index = [리스트]
# 단, 컬럼(딕셔너리의 key)의 데이터(딕셔너리의 value) 갯수와 동일

# index로 사용될 리스트 생성
dataframe_date = ['21.12.01', '21.12.02', '21.12.03', '21.12.04', '21.12.05']

daesin_day3 = DataFrame(daeshin, columns=['open', 'low', 'close', 'high'], index=dataframe_date)

# DataFrame의 데이터 추출 : 컬럼명 또는 index를 이용
print(daesin_day3['open'])

print(daesin_day3['21.12.01':'21.12.01'])
print(daesin_day3['21.12.02':'21.12.04'])


###### pandas 전체 모듈 import 시, Series와 DataFrame 사용방법
###### 숫자 관련 Numpy 모듈의 기능 사용하는 방법

import pandas as pd
import numpy as np

# 숫자가 아닌 데이터(NaN : Not a Number) 삽입 : Numpy의 nan 을 이용
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Pandas 의 date_range('시작날짜', periods=갯수)
dates = pd.date_range('20211208', periods=6)

# Numpy 를 이용하여 행열형태의 난수(임의의 수) 생성 방법 : np.random.randn(행, 열)
np.random.randn(6, 4)

# dates를 인덱스로 np.random.randn(6, 4)를 데이터로 => DataFrame 객체 생성
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['A', 'B', 'C', 'D'] )
'''
Out[38]: df의 내부 데이터
                   A         B         C         D
2021-12-08 -0.674593  0.527925  0.756889  0.482493
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-10  0.235613  0.310132  0.583826  0.693521

2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
2021-12-13  0.405112  0.582745 -1.011934 -1.198881
'''

# DataFrame의 상위 / 하위 일정 데이터만 추출 : 상위 : head() / 하위 : tail()

df.head()
'''
Out[39]: 
                   A         B         C         D
2021-12-08 -0.674593  0.527925  0.756889  0.482493
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-10  0.235613  0.310132  0.583826  0.693521
2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
'''

df.head(3)
'''
Out[40]: 
                   A         B         C         D
2021-12-08 -0.674593  0.527925  0.756889  0.482493
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-10  0.235613  0.310132  0.583826  0.693521
'''

df.tail()
'''
                   A         B         C         D
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-10  0.235613  0.310132  0.583826  0.693521
2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
2021-12-13  0.405112  0.582745 -1.011934 -1.198881
'''

df.tail(2)
'''
                   A         B         C         D
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
2021-12-13  0.405112  0.582745 -1.011934 -1.198881
'''

# DataFrame의 index 만 추출 : index
df.index
'''
DatetimeIndex(['2021-12-08', '2021-12-09', '2021-12-10', '2021-12-11',
               '2021-12-12', '2021-12-13'],
              dtype='datetime64[ns]', freq='D')
'''

# DataFrame의 컬럼명 : columns
df.columns
'''
Out[44]: Index(['A', 'B', 'C', 'D'], dtype='object')
'''

# DataFrame의 데이터 : values
df.values
'''
array([[-0.67459303,  0.52792474,  0.75688937,  0.48249335],
       [ 0.26615252, -1.20258436,  0.01670894, -0.08102408],
       [ 0.23561282,  0.31013201,  0.58382555,  0.69352093],
       [ 0.47894214, -0.09437638,  0.09295792,  0.63227479],
       [ 0.30708005,  0.00124196, -0.35570158, -0.53509624],
       [ 0.40511192,  0.58274461, -1.01193448, -1.19888083]])
'''

# DataFrame의 정보 : info()
df.info()
'''
<class 'pandas.core.frame.DataFrame'>                    <- 타입
DatetimeIndex: 6 entries, 2021-12-08 to 2021-12-13       <- index 객수, 생성날짜
Freq: D
Data columns (total 4 columns):                          <- 컬럼 갯수
 #   Column  Non-Null Count  Dtype                       <- 각 컬럼 정보
---  ------  --------------  -----  
 0   A       6 non-null      float64
 1   B       6 non-null      float64
 2   C       6 non-null      float64
 3   D       6 non-null      float64
dtypes: float64(4)
memory usage: 240.0 bytes                                <- 크기
'''

# DataFrame의 통계 요약 : describe()
df.describe()
'''
              A         B         C         D      <- 컬럼명
count  6.000000  6.000000  6.000000  6.000000      <- 각 컬럼의 데이터 갯수
mean   0.169718  0.020847  0.013791 -0.001119      <- 각 컬럼의 평균
std    0.423415  0.658329  0.643747  0.754154      <- 각 컬럼의 표준편차
min   -0.674593 -1.202584 -1.011934 -1.198881      <- 각 컬럼의 최소값
25%    0.243248 -0.070472 -0.262599 -0.421578      <- 각 컬럼의 1사분위값
50%    0.286616  0.155687  0.054833  0.200735      <- 각 컬럼의 2사분위값
75%    0.380604  0.473477  0.461109  0.594829      <- 각 컬럼의 3사분위값
max    0.478942  0.582745  0.756889  0.693521      <- 각 컬럼의 최대값
'''

# DataFrame의 데이터 정렬 : sort_values(by=기준, ascending=True/False)
df.sort_values(by='B', ascending=False)
'''
                   A         B         C         D
2021-12-13  0.405112  0.582745 -1.011934 -1.198881
2021-12-08 -0.674593  0.527925  0.756889  0.482493
2021-12-10  0.235613  0.310132  0.583826  0.693521
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
'''

df.sort_values(by='B', ascending=True)
'''
                   A         B         C         D
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
2021-12-10  0.235613  0.310132  0.583826  0.693521
2021-12-08 -0.674593  0.527925  0.756889  0.482493
2021-12-13  0.405112  0.582745 -1.011934 -1.198881
'''

# DataFrame의 컬럼명으로 데이터 추출 : DataFrame[컬럼명]
df['A']
'''
2021-12-08   -0.674593
2021-12-09    0.266153
2021-12-10    0.235613
2021-12-11    0.478942
2021-12-12    0.307080
2021-12-13    0.405112
Freq: D, Name: A, dtype: float64
'''

# DataFrame의 내부 index로 데이터 추출 : DataFrame[시작index : 끝index]
df[0 : 3]
'''
                   A         B         C         D
2021-12-08 -0.674593  0.527925  0.756889  0.482493
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-10  0.235613  0.310132  0.583826  0.693521
'''

# DataFrame의 index명으로 데이터 추출 : DataFrame[시작index명 : 끝index명]
df['2021-12-10' : '2021-12-12']
'''
                   A         B         C         D
2021-12-10  0.235613  0.310132  0.583826  0.693521
2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
'''

# DataFrame의 컬럼명과 ndex명으로 데이터 추출 : DataFrame.loc[시작index명 : 끝index명, [컬럼명, 컬럼명] ]
#                                                                         행                  열
df.loc['2021-12-10' : '2021-12-12', ['A', 'B']]
'''
                   A         B
2021-12-10  0.235613  0.310132
2021-12-11  0.478942 -0.094376
2021-12-12  0.307080  0.001242
'''

# DataFrame의 컬럼의 값을 비교하여 추출 : DataFrame[DataFrame.컬럼명   비교연산   비교값]
df[df.A > 0]     # A 컬럼의 데이터가 0보다 큰 값들에 해당하는 모든 컬럼값
'''
                   A         B         C         D
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-10  0.235613  0.310132  0.583826  0.693521
2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
2021-12-13  0.405112  0.582745 -1.011934 -1.198881
'''

# DataFrame을 복사 : DataFrame.copy()
df2 = df.copy()
'''
df2
Out[56]: 
                   A         B         C         D
2021-12-08 -0.674593  0.527925  0.756889  0.482493
2021-12-09  0.266153 -1.202584  0.016709 -0.081024
2021-12-10  0.235613  0.310132  0.583826  0.693521
2021-12-11  0.478942 -0.094376  0.092958  0.632275
2021-12-12  0.307080  0.001242 -0.355702 -0.535096
2021-12-13  0.405112  0.582745 -1.011934 -1.198881
'''

# 기존 DataFrame에 새로움 컬럼과 데이터 추가 : DataFrame[새로운 컬럼명] = [신규데이터들]
df2['E'] = ['one', 'two', 'three', 'four', 'one', 'two']
'''
df2
Out[58]: 
                   A         B         C         D      E
2021-12-08 -0.674593  0.527925  0.756889  0.482493    one
2021-12-09  0.266153 -1.202584  0.016709 -0.081024    two
2021-12-10  0.235613  0.310132  0.583826  0.693521  three
2021-12-11  0.478942 -0.094376  0.092958  0.632275   four
2021-12-12  0.307080  0.001242 -0.355702 -0.535096    one
2021-12-13  0.405112  0.582745 -1.011934 -1.198881    two
'''

# DataFrame의 특정 컬럼에 지정한 데이터가 포함되어 있는지 확인 : DataFrame[컬럼명].isin([조회데이터])
df2['E'].isin(['two', 'three'])
'''
Out[59]: 
2021-12-08    False   <- 'two' 또는 'three' 포함하지 않음
2021-12-09     True   <- 'two' 또는 'three' 포함
2021-12-10     True   <- 'two' 또는 'three' 포함
2021-12-11    False   <- 'two' 또는 'three' 포함하지 않음
2021-12-12    False   <- 'two' 또는 'three' 포함하지 않음
2021-12-13     True   <- 'two' 또는 'three' 포함
Freq: D, Name: E, dtype: bool
'''

# 전체 DataFrame으로부터 DataFrame의 특정 컬럼에 지정한 데이터가 포함되어 있는 데이터만 추출
df2[df2['E'].isin(['two', 'three'])]
'''
                   A         B         C         D      E
2021-12-09  0.266153 -1.202584  0.016709 -0.081024    two
2021-12-10  0.235613  0.310132  0.583826  0.693521  three
2021-12-13  0.405112  0.582745 -1.011934 -1.198881    two
'''



####### DataFrame 병합 : 두 개 이상의 DataFrame 합하기 
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

# DataFrame들을 순서대로 연결하여 새로운 DataFrame으로 생성 : Pandas.concat([데이터프레임, 데이터프레임...])
result = pd.concat([df1, df2, df3])
'''
Out[63]: 
      A    B    C    D
0    A0   B0   C0   D0
1    A1   B1   C1   D1
2    A2   B2   C2   D2
3    A3   B3   C3   D3

4    A4   B4   C4   D4
5    A5   B5   C5   D5
6    A6   B6   C6   D6
7    A7   B7   C7   D7

8    A8   B8   C8   D8
9    A9   B9   C9   D9
10  A10  B10  C10  D10
11  A11  B11  C11  D11
'''

# DataFrame 연결시 레벨에 해당하는 key를 부여할 경우 : keys=[리스트]
result = pd.concat([df1, df2, df3], keys=['x', 'y', 'z'])
'''
Out[65]: 
        A    B    C    D
x 0    A0   B0   C0   D0
  1    A1   B1   C1   D1
  2    A2   B2   C2   D2
  3    A3   B3   C3   D3
y 4    A4   B4   C4   D4
  5    A5   B5   C5   D5
  6    A6   B6   C6   D6
  7    A7   B7   C7   D7
z 8    A8   B8   C8   D8
  9    A9   B9   C9   D9
  10  A10  B10  C10  D10
  11  A11  B11  C11  D11
'''

# DataFrame 연결시 행이 아닌 열로 연결하고 싶을 경우 : axis=1 로 설정
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'], 
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])

result = pd.concat([df1, df4])   # 행기준
'''
     A   B    C   D    F
0   A0  B0   C0  D0  NaN
1   A1  B1   C1  D1  NaN
2   A2  B2   C2  D2  NaN
3   A3  B3   C3  D3  NaN
2  NaN  B2  NaN  D2   F2
3  NaN  B3  NaN  D3   F3
6  NaN  B6  NaN  D6   F6
7  NaN  B7  NaN  D7   F7
'''

result = pd.concat([df1, df4], axis=1)  # 열기준
'''
Out[71]: 
     A    B    C    D    B    D    F
0   A0   B0   C0   D0  NaN  NaN  NaN
1   A1   B1   C1   D1  NaN  NaN  NaN
2   A2   B2   C2   D2   B2   D2   F2
3   A3   B3   C3   D3   B3   D3   F3
6  NaN  NaN  NaN  NaN   B6   D6   F6
7  NaN  NaN  NaN  NaN   B7   D7   F7
'''

# DataFrame 연결시 공통 부분만 : join='inner'
result = pd.concat([df1, df4], axis=1, join="inner") 
'''
Out[73]: 
    A   B   C   D   B   D   F
2  A2  B2  C2  D2  B2  D2  F2
3  A3  B3  C3  D3  B3  D3  F3
'''

#  DataFrame 연결시 기존 index를 무시하고 다시 index를 0부터 정수로 재설정 할 경우 : ignore_index=True
result = pd.concat([df1, df4], ignore_index=True)
'''
Out[75]: 
     A   B    C   D    F
0   A0  B0   C0  D0  NaN
1   A1  B1   C1  D1  NaN
2   A2  B2   C2  D2  NaN
3   A3  B3   C3  D3  NaN
4  NaN  B2  NaN  D2   F2
5  NaN  B3  NaN  D3   F3
6  NaN  B6  NaN  D6   F6
7  NaN  B7  NaN  D7   F7
'''


# DataFrame 병합 : Pandas.merge(데이터프레임, 데이터프레임, on=key이름, how="병합방식")
left = pd.DataFrame({'key': ['K0', 'K4', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

pd.merge(left, right, on='key')
'''
Out[77]: 
  key   A   B   C   D
0  K0  A0  B0  C0  D0
1  K2  A2  B2  C2  D2
2  K3  A3  B3  C3  D3
'''

pd.merge(left, right, how='left', on='key')
'''
Out[78]: 
  key   A   B    C    D
0  K0  A0  B0   C0   D0
1  K4  A1  B1  NaN  NaN
2  K2  A2  B2   C2   D2
3  K3  A3  B3   C3   D3
'''

pd.merge(left, right, how='right', on='key')
'''
Out[79]: 
  key    A    B   C   D
0  K0   A0   B0  C0  D0
1  K1  NaN  NaN  C1  D1
2  K2   A2   B2  C2  D2
3  K3   A3   B3  C3  D3
'''

pd.merge(left, right, how='outer', on='key')
'''
Out[80]: 
  key    A    B    C    D
0  K0   A0   B0   C0   D0
1  K4   A1   B1  NaN  NaN
2  K2   A2   B2   C2   D2
3  K3   A3   B3   C3   D3
4  K1  NaN  NaN   C1   D1
'''

pd.merge(left, right, how='inner', on='key')
'''
  key   A   B   C   D
0  K0  A0  B0  C0  D0
1  K2  A2  B2  C2  D2
2  K3  A3  B3  C3  D3
'''



#### 외부파일을 DataFrame 형태로 불러오기 : Pandas.read_csv() / Pandas.read_excel()
### 파일위치 : 파이썬 파일의 하위폴더인 data1 내부에 읽기 위한 파일이 존재 : ./data1/~~ 와 같은 상대경로를 사용
### 작업에 필요한 모듈 impport
import pandas as pd
import numpy as np

### 1. 01. CCTV_in_Seoul.csv 파일 읽기
cctv = pd.read_csv("./data1/01. CCTV_in_Seoul.csv")

# 컬럼명 확인
cctv.columns
'''
Out[86]: Index(['기관명', '소계', '2013년도 이전', '2014년', '2015년', '2016년'], dtype='object')
'''

cctv.columns[0]
'''
Out[87]: '기관명'
'''

# 컬럼명 변경 : 기관명 => 구별 : DataFrame.rename(columns={DataFrame.columns[0] : '변경컬럼영'}, inplace=True
# inplace=True : 변경사항을 해당 데이터프레임에 적용
cctv.rename(columns={cctv.columns[0] : '구별'}, inplace=True)


### 2. 01. population_in_Seoul.xls 파일 읽기
# 전체 읽기
seoul = pd.read_excel("./data1/01. population_in_Seoul.xls")

# 행 건너뛰기 : header=엑셀행번호
seoul = pd.read_excel("./data1/01. population_in_Seoul.xls", header=2)

# 열 선택하기 : usecols='엑셀컬럼명, 엑셀컬럼명, 엑셀컬럼명,...'
seoul = pd.read_excel("./data1/01. population_in_Seoul.xls", header=2, usecols='B, D, G, J, N')




