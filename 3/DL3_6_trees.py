import numpy as np
import tensorflow as tf


trees=np.loadtxt('d:/dl/data/trees.csv',dtype=np.float32,skiprows=1,delimiter=",")
# print(trees.shape) #(3,31)
xdata=trees[:,0:-1] #x=[[둘레],[높이]]
ydata=trees[:,[-1]] #[볼륨]
print(xdata)
print(ydata)

x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random_normal([2,1]))
b=tf.Variable(tf.random_normal([1]))
# w(1,2)* x(2,31) = hf(1,31)
hf=tf.matmul(w,x)+b

# 트레이닝을 50회수행 => 둘레:13, 높이:90, 볼륨?예측
cost=tf.reduce_mean((hf-y)**2)
optimizer=tf.train.GradientDescentOptimizer(0.00015)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(51):
    cv,hv,_=sess.run([cost, hf,train],feed_dict={x:xdata,y:ydata})
    if step%10==0:
        print(step,cv)
sess.close()