import numpy as np
import tensorflow as tf
xy=np.loadtxt('d:/dl/data/zoo.csv',delimiter=",",dtype=np.float32)
xdata = xy[:,0:-1]
ydata = xy[:,[-1]]
# print(xdata.shape,ydata.shape)
nb_classes=7 #0~6
x=tf.placeholder(tf.float32,[None,16])
y=tf.placeholder(tf.int32,[None,1])

#y에는 0~6사이의 임의의 수 저장
#원핫 인코딩 해야함.
y_one_hot =tf.one_hot(y, nb_classes)
print("one hot 상태: ",y_one_hot)
#0 -> 1000000, 3 -> 0001000
#원핫 인코딩을 수행하면 차원이 1 증가
#예를 들어 y 가 (None,1) ->(None,1,7)이 됨
#[[0],[3]]->[[[1000000]],[[0001000]]]
y_one_hot=tf.reshape(y_one_hot,[-1,nb_classes]) #--> -1은 전체 데이터를 정의
print("reshape 결과: ",y_one_hot)

w=tf.Variable(tf.random_normal([16,nb_classes]))
b=tf.Variable(tf.random_normal([nb_classes]))
logits =tf.matmul(x,w)+b #logit=score
hf=tf.nn.softmax(logits)

cost_i =tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

prediction=tf.argmax(hf,axis=1)
correct_prediction=tf.equal(prediction,tf.argmax(y_one_hot,axis=1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={x:xdata,y:ydata})
        if step%100==0:
            cv,av=sess.run([cost,accuracy],feed_dict={x:xdata,y:ydata})
            print(step, cv, av)
    print(sess.run(prediction,feed_dict={x:[[0.,0.,1.,0.,0.,
                                       1.,1.,1.,1.,0.,
                                       0.,1.,0.,1.,0.,0.]]}))