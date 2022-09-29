from telnetlib import X3PAD
import tensorflow as tf

텐서 = tf.constant([3,4,5])
텐서2 = tf.constant([7,8,9])

print(tf.add(텐서,텐서2))
#tf.add
#tf.substract
#tf.divide
#tf.multiply

tf.zeros([2,2,3]) # 0이라는 데이터를 3개담은 리스트 2개를 담은 2개의 리스트, 뒤에서부터 해석

#print(변수명.shape) => (행,열), 열개의 데이터가 있는 행개의 리스트가 있다.

w = tf.Variable(1.0) #1을 가진 텐서 weight는 variable로
w.assign(2) #=> 2로 값 변경
