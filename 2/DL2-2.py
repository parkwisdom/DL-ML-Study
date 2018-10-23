#linear regression

import tensorflow as tf
tf.set_random_seed(777)

x_train=[1,2,3]
y_train=[1,2,3]
# #data-set -> training(7) :test(3)
# ex) data 100개중 트레이닝 70개, 테스트30개 사용

w=tf.Variable(tf.random_normal([1])) #정규분포 난수 1개 -변수
b=tf.Variable(tf.random_normal([1])) #정규분포 난수 1개 -변수
hf = x_train*w+b

cost=tf.reduce_mean(tf.square(hf-y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)
##########▲그래프 정의

sess=tf.Session()
sess.run(tf.global_variables_initializer()) # 초기화

for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(cost),sess.run(w),sess.run(b))

