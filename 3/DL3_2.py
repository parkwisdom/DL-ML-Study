import tensorflow as tf
import numpy as np


# def not_used():
# #Muti-variable regression
#     x1 = [1, 0, 3, 0, 5]
#     x2 = [0, 2, 0, 4, 0]
#     y = [1, 2, 3, 4, 5]
#     w1 = tf.Variable(tf.random_uniform([1], -1, 1))
#     w2 = tf.Variable(tf.random_uniform([1], -1, 1))
#     b = tf.Variable(tf.random_uniform([1], -1, 1))
#
#     hf = w1 * x1 + w2 * x2 + b
#     cost = tf.reduce_mean((hf - y) ** 2)
#     optimixer = tf.train.GradientDescentOptimizer(0.1)
#     train = optimixer.minimize(cost)
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(10):
#         sess.run(train)
#     sess.close()
# not_used()
#
# def not_used_2():
#     x=[[1,0,3,0,5],
#        [0,2,0,4,0]]
#     y=[1,2,3,4,5]
#     w=tf.Variable(tf.random_uniform([1,2],-1,1))
#     b = tf.Variable(tf.random_uniform([1], -1, 1))
#
#     # (1,2)*(2,5) -->행렬곱..
#     # w  *  x =hf
#
#     hf = tf.matmul(w,x)+b
#     cost = tf.reduce_mean(tf.square(hf-y))
#     optimixer = tf.train.GradientDescentOptimizer(0.1)
#     train = optimixer.minimize(cost)
#
#     sess=tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(10):
#         sess.run(train)
#     sess.close()
# not_used_2()

# def not_used_3():
#     x = [[1, 1, 1, 1, 1],
#          [1, 0, 3, 0, 5],
#          [0, 2, 0, 4, 0]]
#     y = [1, 2, 3, 4, 5]
#     w = tf.Variable(tf.random_uniform([1,3], -1, 1))
#     b = tf.Variable(tf.random_uniform([1], -1, 1))
#
#
#     hf = tf.matmul(w,x)+b
#     cost = tf.reduce_mean((hf - y) ** 2)
#     optimizer = tf.train.GradientDescentOptimizer(0.1)
#     train = optimizer.minimize(cost)
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(10):
#         sess.run(train)
#     sess.close()

######★★★수정 필요★★★
def not_used_3():
    trees=np.loadtxt('data/trees.csv',unpack=True, skiprows=1,delimiter=",")
    print(trees.shape) #(3,31)
    x=[trees[0],trees[1]] #x=[[둘레],[높이]]
    y=trees[-1] #[볼륨]
    #print(y)
    w=tf.Variable(tf.random_uniform([1,2],-1,1))
    # w(1,2)* x(2,31) = hf(1,31)
    hf=tf.matmul(w,x)
    # cost=
    # optimizer=    #0.00015
    # train=
    # 트레이닝을 50회수행 =>
    # 둘레:13, 높이:90, 볼륨?예측
########################33

not_used_3()


