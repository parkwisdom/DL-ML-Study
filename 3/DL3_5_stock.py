import numpy as np
import tensorflow as tf

stock = np.loadtxt("D:/DL/data/Alphabet Inc stock.csv",skiprows=1,delimiter=",",dtype=np.float32)
# print(stock)
xdata = stock[:500,0:-1]
ydata = stock[:500,[-1]]

x= tf.placeholder(tf.float32,shape=[None,3])
y= tf.placeholder(tf.float32,shape=[None,1])
w=tf.Variable(tf.random_normal([3,1]))
b=tf.Variable(tf.random_normal([1]))
hf = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hf-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=2e-7)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cv,hv,_=sess.run([cost,hf,train],feed_dict={x:xdata,y:ydata})
    # if step%1000==0:
    #     print(cv,hv)

print("예측값↓ \t" ,"\t원본값↓")
print(np.concatenate((hv,ydata),axis=1))

#
# 예측값↓ 	 	원본값↓
# [[339.14066 337.97678]
#  [337.01727 336.46164]
#  [336.29968 336.69016]
#  [336.4499  335.3936 ]
#  [336.64987 337.95193]
#  [341.34326 341.5734 ]
#  [342.89844 341.73734]
#  [337.62402 339.27335]
# .
# .
# .
#  [574.7073  574.19275]
#  [564.1988  560.6572 ]
#  [562.1441  569.3693 ]
#  [559.4129  557.81287]
#  [549.05536 541.51245]
#  [536.06323 530.2941 ]
#  [537.49744 534.9983 ]
#  [519.7482  527.13153]
#  [520.4157  521.6417 ]
#  [512.1377  508.3747 ]
#  [514.2745  517.9918 ]]
