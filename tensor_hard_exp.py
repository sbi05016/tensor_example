import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')
print(data)

print(data.isnull().sum()) #null 값 세줌
data = data.dropna()
y데이터 = data['admit'].values #리스트에 담아줌 # 값데이터
x데이터 = []
for i,rows in data.iterrows(): #한행 씩 출력 #i 행번호 #rows 내용
  x데이터.append([rows['gre'],rows['gpa'],rows['rank']])
  

import tensorflow as tf

model = tf.keras.models.Sequential([   # 딥러닝 모델 만드는
  tf.keras.layers.Dense(64,activation='tanh'),
  #레이어 만드는 ()개수, 안에 값은 관습적으로 2의 제곱수 가장 알맞은 값을 때려맞춰서 찾아야함
  #activate_function : sigmoid,tanh,relu,softmax,leakyRelu
  tf.keras.layers.Dense(128,activation='tanh'),
  tf.keras.layers.Dense(1,activation='sigmoid'), 
  # 마지막에 올꺼는 1개
  # 마지막 레이어는 예측결과를 뱉어야함
  #0~1사이의 확률을 뱉고 싶으면 sigmoid ㄱㄱ
  ]) 

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# 경사하강법으로 w값을 수정할때 기울기 값 만큼 빼서 수정하는데 이때 빼는 값을 optimiazer가 조정해줌
# optimizer 종류로는 adam,adagrad,adadelta,rmsprop,sgd

#loss
#확률 예측,0~1분류 문제 => binary_crossentropy
model.fit(np.array(x데이터),np.array(y데이터), epochs = 1000)
# 값 넣을 때 numpy array나 tensor를 넣어야한다.
#학습시키기 #fit(트레이닝데이터(정답예측에 필요한 인풋),값(정답),epochs=몇번 학습시킬지)

#예측
pre = model.predict([[750, 3.70, 3],[400,2.2,1]])
print(pre)
