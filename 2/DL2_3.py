#linear regression

import tensorflow as tf
tf.set_random_seed(777)

w=tf.Variable(tf.random_normal([1])) #정규분포 난수 1개 -변수
b=tf.Variable(tf.random_normal([1])) #정규분포 난수 1개 -변수
x= tf.placeholder(tf.float32, shape=[None])
y= tf.placeholder(tf.float32, shape=[None])

hf = x*w+b

cost=tf.reduce_mean(tf.square(hf-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)
##########▲그래프 정의

sess=tf.Session()
sess.run(tf.global_variables_initializer()) # 초기화

for step in range(2001):
    sess.run(train,feed_dict={x:[1,2,3],y:[1,2,3]}) #그래프 실행
    if step%20==0:
        print(step,sess.run(cost,feed_dict={x:[1,2,3],y:[1,2,3]}))
print("---------모델완성-----------")
# print(step,sess.run(cost,feed_dict={x:[1,2,3],y:[1,2,3]}))
# print(step,sess.run(w))
# print(step,sess.run(b))
cv,wv,bv=sess.run([cost,w,b],feed_dict={x:[1,2,3],y:[1,2,3]})
print("비용:",cv, "가중치:",wv, "편향:",bv)


#예측해보기
#x=2.5 ,x=5, x=1.5, 3.5???
print("2.5일때 예상되는 값: ",sess.run(hf,feed_dict={x:[2.5]}))
print("5일때 예상되는 값: ",sess.run(hf,feed_dict={x:[5]}))
print("1.5일때와 3.5일때 예상되는 값: ",sess.run(hf,feed_dict={x:[1.5,3.5]}))

