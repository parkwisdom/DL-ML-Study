import tensorflow as tf
import numpy as np
import csv
tf.set_random_seed(777)

read=open("d:/dl/data/iris.csv","r",encoding="utf-8")
# print(read)
csvread=csv.reader(read)
# print(csvread) #<_csv.reader object at 0x0000014153D708D0>
next(csvread) #첫번째 줄 skip(한줄씩 skip함)

xdata=[]
ydata=[]

index =['setosa','versicolor','virginica']
for row in csvread:
    # print(row)
    data=[]
    sepal_length =float(row[1])
    sepal_width =float(row[2])
    sepal_length =float(row[3])
    sepal_width =float(row[4])
    data=[sepal_length,sepal_width,sepal_length,sepal_width]
    xdata.append(data)
    # print(xdata)

    for i in range(3):
        if row[5]==index[i]:
            ydata.append([i])

print(xdata)
print(ydata)

x=tf.placeholder(tf.float32,shape=[None,4])
y=tf.placeholder(tf.int32,shape=[None,1])
w=tf.Variable(tf.random_normal([4,3]))
b=tf.Variable(tf.random_normal([3]))

nb_classes=3
y_one_hot =tf.one_hot(y,nb_classes)
y_one_hot=tf.reshape(y_one_hot,[-1,nb_classes])
# [[0],[2]] -one hot-> [[[100]],[[001]]] -reshape->[[100],[001]]

logit =tf.matmul(x,w)+b
hf=tf.nn.softmax(logit)

costi=tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=y_one_hot)
cost= tf.reduce_mean(costi)

prediction = tf.argmax(hf,1)
corrent_prediction=tf.equal(prediction,tf.argmax(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(corrent_prediction,dtype=tf.float32))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
## ▲그래프 정의---------------------------------

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cv,av,_=sess.run([cost,accuracy,train],feed_dict={x:xdata,y:ydata})
    if step%20==0:
        print('cost:',cv,'acc:',av)

pred=sess.run(prediction, feed_dict={x:xdata})
ydata = np.array(ydata,dtype=np.int32)
for p,y in zip(pred, ydata.flatten()): #flatten 함수는 np에 있음. ydata 의 구조를 리스트->array 로 변경
    print("{} prediction: {} True Y: {}".format(p==y,p,y))