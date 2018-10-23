import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

cars=np.loadtxt('cars.csv',delimiter=',',unpack=True)
# print(cars.shape)
# print(cars)
# x=cars[0] #스피드
# y=cars[1] #제동거리
# print(x,y)

w=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))
x=tf.placeholder(tf.float32,shape=[None])
y=tf.placeholder(tf.float32,shape=[None])

hf = x*w+b

cost = tf.reduce_mean(tf.square(hf-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    sess.run(train,feed_dict={x:cars[0],y:cars[1]})
    if step%100==0:
        print(step, sess.run(cost,feed_dict={x:cars[0],y:cars[1]}))
print("-----------모델완성---------")

print("30일때 예상되는 값: ",sess.run(hf,feed_dict={x:[30]}))
print("50일때 예상되는 값: ",sess.run(hf,feed_dict={x:[50]}))
