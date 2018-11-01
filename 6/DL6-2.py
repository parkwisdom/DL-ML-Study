import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]  #(8,4)
y_data = [[0, 0, 1],#2
          [0, 0, 1],#2
          [0, 0, 1],#2
          [0, 1, 0],#1
          [0, 1, 0],#1
          [0, 1, 0],#1
          [1, 0, 0],#0
          [1, 0, 0]]#0  #(8,3)

"""
x,y=플레이스홀더
클래스 개수=3
w,b=변수(랜덤값 초기화)
learning_rate = 0.1
트레이닝 횟수:2001
"""
#x값이 [[1,11,7,9],[1,3,4,3],[1,1,0,1]] 일 때 분류 결과를 출력하시오.


x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
w=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))

z=tf.matmul(x,w)
hf=tf.nn.softmax(z)
cost = tf.reduce_mean(tf.reduce_sum(y*-tf.log(hf),axis=1))
train=tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x:x_data,y:y_data})
    if i%20==0:
        print(i,sess.run(cost, feed_dict={x:x_data,y:y_data}))

xhat = sess.run(hf, feed_dict={x:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
print("결과:",xhat)

# grades= ['a','b','c']
# print(sess.run(tf.argmax(xhat,axis=1)))
# xhat2=sess.run(tf.argmax(xhat,axis=1))
# # print(grades[xhat2[0]],grades[xhat2[1]],grades[xhat2[2]])

