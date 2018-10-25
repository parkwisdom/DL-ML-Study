import math
import numpy as np
import tensorflow as tf

##########당뇨병 훈련&테스트############
diabetes = np.loadtxt("d:/dl/data/diabetes.csv",delimiter=',')
# print(diabetes.shape) #(759, 9)
#trainingData
xdata = diabetes[:500,0:-1]
ydata = diabetes[:500,[-1]]

#testData
xtest = diabetes[501:,0:-1]
ytest =diabetes[501:,[-1]]

x=tf.placeholder(tf.float32,shape=[None,8])
y=tf.placeholder(tf.float32,shape=[None,1])
w=tf.Variable(tf.random_normal([8,1]))
b=tf.Variable(tf.random_normal([1]))
hf=tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y*tf.log(hf)+(1-y)*tf.log(1-hf))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

predicted = tf.cast(hf>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cv,_ = sess.run([cost,train],feed_dict={x:xdata,y:ydata})
        # if step%1000==0:
        #     print(step,cv)

    hft,pt,at = sess.run([hf,predicted,accuracy],feed_dict={x:xdata,y:ydata})
    # print("\n훈련예측값:",hft,"\n훈련예측값(0/1):",pt,"\n훈련정확도:",at)
    print("\n훈련정확도:",at)

    hfv,pv,av = sess.run([hf,predicted,accuracy],feed_dict={x:xtest,y:ytest})
    # print("\n예측값:",hfv,"\n예측값(0/1):",pv,"\n정확도:",av)
    print("\ntest정확도:",av)

#결과--------------
# 훈련정확도: 0.76
#
# test정확도: 0.79457366
