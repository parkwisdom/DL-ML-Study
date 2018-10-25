#Logistic Classification
import numpy as np
import tensorflow as tf
import math

xx=[[1,1,1,1,1,1],
    [2,3,3,5,7,2],
    [1,2,5,5,5,5]]
yy=np.array([0,0,0,1,1,1])

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
w=tf.Variable(tf.random_uniform([1,3],-1,1))
z=tf.matmul(w,x)
#(1,3)*(3,6)=>(1,6)
hf=tf.nn.sigmoid(z) # hf=1/(1+tf.exp(-z)) #hf = tf.div(1,1+tf.exp(-z))

cost = -tf.reduce_mean(y*tf.log(hf)+(1-y)*tf.log(1-hf))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x:xx,y:yy})
    if i%20==0:
        print(i, sess.run(cost, feed_dict={x:xx,y:yy}))
print("-"*50)
xx=[[1,1],
    [3,7],
    [8,2]]
print(sess.run(hf,feed_dict={x:xx}))
y_hat=sess.run(hf,feed_dict={x:xx})
print(y_hat>0.5)
print(sess.run(w))

def sigmoid(z):
    return 1/(1+math.e**-z)
ww=sess.run(w)
z=np.dot(ww,xx)
print(sigmoid(z))

tf.set_random_seed(777)
xdata = [[1,2],
         [2,3],
         [3,1],
         [4,3],
         [5,3],
         [6,2]]
ydata=[[0],
       [0],
       [0],
       [1],
       [1],
       [1]]

x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])
w=tf.Variable(tf.random_normal([2,1]))
b=tf.Variable(tf.random_normal([1]))
hf = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y*tf.log(hf)+(1-y)*tf.log(1-hf))
train=tf.train.GradientDescentOptimizer(0.01).minimize(cost)

#정확도 계산
predicted=tf.cast(hf>0.5,dtype=tf.float32) #hf>0.5보다 크면 1로 작으면 0으로 표시
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32)) #y와 predicted의 값을 비교하여 정확도


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cv,_=sess.run([cost, train],feed_dict={x:xdata,y:ydata})
        if step%200==0:
            print(step, cv)
    hfv,pv,av=sess.run([hf,predicted,accuracy],feed_dict={x:xdata,y:ydata})
    print("\n예측값:",hfv,"\n예측값(0/1):",pv, "\n정확도:",av)