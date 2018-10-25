import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# x_data = [[280,310,270],
#           [310,300,325],
#           [250,225,205],
#           [270,267,290],
#           [277,307,300]]
#
# y_data = [[299],[327],[240],[270],[290]] #수능점수
#
# x=tf.placeholder(tf.float32,shape=[None,3])
# y=tf.placeholder(tf.float32,shape=[None,1])
# w=tf.Variable(tf.random_normal([3,1]))
# b=tf.Variable(tf.random_normal([1]))
# hf=tf.matmul(x,w)+b
#
# cost = tf.reduce_mean(tf.square(hf-y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=2e-6) #0.00001
# train = optimizer.minimize(cost)
#
# #세션 실행#
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
#
# #학습
# for step in range(10001):
#     cv,hv,_= sess.run([cost, hf, train],feed_dict={x:x_data,y:y_data})
#     if step%100==0:
#         print(step, "비용:",cv,"\n예측:\n",hv)



xy=np.loadtxt("d:/DL/data/score.csv",delimiter=',',dtype=np.float32)
# print(xy)
xdata = xy[ : , 0:-1]
# print(xdata)
ydata= xy[ : , [-1]]
# print(ydata)

x = tf.placeholder(tf.float32,shape=[None,3])
y = tf.placeholder(tf.float32,shape=[None,1])
w = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))
hf = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hf-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cv,hv,_= sess.run([cost, hf,train],feed_dict={x:xdata,y:ydata})

print("예측값:",hv.T ,"\n원본값:",ydata.T )

# 예측값: [[156.92554  184.32603  180.98457  198.38837  139.8508   105.385666
#   150.29037  113.84598  174.15103  164.3553   143.76323  142.7234
#   185.62877  152.56264  151.28001  188.20667  144.03113  180.86986
#   176.62169  158.20985  175.83972  174.17613  167.20522  150.71817
#   190.22943 ]]
# 원본값: [[149. 185. 180. 196. 142. 101. 149. 115. 175. 164. 141. 141. 184. 152.
#   148. 192. 147. 183. 177. 159. 177. 175. 175. 149. 192.]]

