import tensorflow as tf
import numpy as np

xy=np.loadtxt("d:/dl/data/iris_softmax1029.csv",delimiter=',',dtype=np.float32)
# print(xy)
xdata = xy[:, 1:-3]
ydata = xy[:, -3:]
ydata= np.array(ydata, dtype=np.int32) #실수를 정수로 바꿔서 출력 float32->int32
# print(xdata)
# print(ydata)

x=tf.placeholder(tf.float32,shape=[None,4])
y=tf.placeholder(tf.int32,shape=[None,3])
w=tf.Variable(tf.random_normal([4,3]))
b=tf.Variable(tf.random_normal([3]))

logit = tf.matmul(x,w)+b
hf= tf.nn.softmax(logit)

costi = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=y)
cost = tf.reduce_mean(costi)

prediction = tf.argmax(hf,axis=1) #0,1,2 중 값이 들어감
correct_prediction = tf.equal(prediction,tf.argmax(y,axis=1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
train = tf.train.AdamOptimizer(0.1).minimize(cost) #GradientDescentOptimizer :기본optimizer, Adam은 성능향상된 optimizer

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    costv, accv, _ =sess.run([cost,accuracy,train], feed_dict={x:xdata,y:ydata})
    if step%20==0:
        print(step,"cost: ", costv,"acc: ",accv)

pred=sess.run(prediction, feed_dict={x:xdata})
ydata = sess.run(tf.argmax(ydata,1))
for p,y in zip(pred,ydata):
    print("{} prediction:{} True Y:{}".format(p==y,p,y))