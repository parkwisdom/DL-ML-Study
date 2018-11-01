import numpy as np
import tensorflow as tf


#-----------------
# xy=np.loadtxt("d:/dl/data/softmax.txt",dtype=np.float32,encoding='utf-8')
# print(xy)
# xx= xy[:,0:3] #입력
# y=  xy[:,3:] #출력
# # print(xx) #(8,3)
# # print(y) #(8,3)
# x=tf.placeholder(tf.float32)
# w=tf.Variable(tf.zeros([3,3]))  #[x개수, y개수]
# # b=tf.Variable()
# #(8,3)*(3,3) = (8,3)
# z=tf.matmul(x,w)
# hf=tf.nn.softmax(z) #z값이 확률로 출력
# cost =tf.reduce_mean(tf.reduce_sum(y*-tf.log(hf),axis=1)) # 열단위로 합쳐서..
# optimizer =tf.train.GradientDescentOptimizer(0.1)
# train = optimizer.minimize(cost)
#
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(2001):
#     sess.run(train, feed_dict={x:xx})
#     if i %20 ==0:
#         print(sess.run(cost,feed_dict={x:xx}))
# #예측 : xrkqtdmfh 1,11,7=>
# #                 1,3,4,=>
# yhat=sess.run(hf,feed_dict={x:[[1,11,7],[1,3,4]]})
# print("예측",yhat)
#
# grades = ['A','B','C']
# print(sess.run(tf.argmax(yhat,axis=1))) #각 행에서 가장 큰 수의 위치
# yhat2 =sess.run(tf.argmax(yhat,axis=1)) #[0 2]
# print(grades[yhat2[0]],grades[yhat2[1]])
#
# grades= np.array(['A','B','C'])
# print(grades[yhat2])
# #---------------------------------------------------

# import math
# def softmax():
#     aa=math.e**2.0
#     bb=math.e**1.0
#     cc=math.e**0.1
#     base=aa+bb+cc
#     print(aa/base)
#     print(bb/base)
#     print(cc/base)
# softmax()
## ---------------------------

def read_iris_softmax():
    iris = np.loadtxt("d:/dl/data/iris_softmax1029.csv",delimiter=',')
    # print(iris.shape)

    # train_set =iris[:40]+iris[50:90]+iris[100:140] #0-40,50-90,100-140 까지의 행들이 더해짐--이렇게 하면 안됨
    # print(train_set)

    train_set=np.vstack((iris[:40],iris[50:90],iris[100:140]))
    # print(train_set)
    # print(train_set.shape)
    test_set=np.vstack((iris[40:50],iris[90:100],iris[140:]))
    # print(test_set.shape)
    return train_set,test_set
train_set,test_set=read_iris_softmax()

xx=train_set[:,:-3]
print(xx.shape) #(120, 5)
y=train_set[:,-3:]
print(y.shape) #(120, 3)

x=tf.placeholder(tf.float32)
w=tf.Variable(tf.zeros([5,3]))
#(120,3)=(120,5)*(5,3)
z=tf.matmul(x,w)
hf=tf.nn.softmax(z)
cost = tf.reduce_sum(tf.reduce_sum(y*-tf.log(hf),axis=1))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train,feed_dict={x:xx})
    if i%20==0:
        print(i, sess.run(cost,feed_dict={x:xx}))

#test 데이터 셋을 이용한 모델 검증
xx=test_set[:,:-3]
y=test_set[:,-3:]

yhat=sess.run(hf, feed_dict={x:xx})
# print(yhat)
# yhat2=sess.run(tf.argmax(yhat,axis=1))
# print("예측값:",yhat2)
# y2=sess.run(tf.argmax(y,axis=1))
# print("실제값:",y2)
#
# #비교
# equal=sess.run(tf.equal(yhat2,y2))
# print(equal)
#
# #수치
# cast= sess.run(tf.cast(equal,tf.float32))
# print(cast)
# mean=sess.run(tf.reduce_mean(cast))
# print(mean)

## np사용하는 방법 ▼
yhat2=np.argmax(yhat,axis=1)
print(yhat2)
y2=np.argmax(y,axis=1)
print(y2)

equal = (yhat2==y2)
print(equal)
mean=np.mean(equal)  #numpy사용하면 casting을 안해줘도 같은 결과 나옴.
print(mean)

