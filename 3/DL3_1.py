import tensorflow as tf
import matplotlib.pyplot as plt

###########placeholder사용 연산
# a=tf.placeholder(tf.int32)
# b=tf.placeholder(tf.int32)
# add=tf.add(a,b)
#
# #a dp 3,b에 4를 전다한 후 합을 출력하는 코드 작성
#
# sess=tf.Session()
# print(sess.run(add,feed_dict={a:3,b:4}))

###############Variable사용 연산
# a= tf.Variable(3)
# b= tf.Variable(4)
# add=tf.add(a,b)
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(add))


# #placeholder를 사용하여 구구단 출력
# def show99(dan):
#     left =tf.placeholder(tf.int32)
#     right = tf.placeholder(tf.int32)
#     op = tf.multiply(left,right)
#     sess = tf.Session()
#
#     for i in range(1,10):
#         res=sess.run(op,feed_dict={left:dan,right:i})
#         print('{} * {} = {}'.format(dan,i,res))
# show99(2)


# #w와 b에 대한 초기값을 부여한 상태에서 모델링
# w=tf.Variable([.3],tf.float32)
# b=tf.Variable([-.3],tf.float32)
# x=tf.placeholder(tf.float32)
# y=tf.placeholder(tf.float32)
# lm=x*w+b
# loss=tf.reduce_sum(tf.square(lm-y))
#
# train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# x_train=[1,2,3,4]
# y_train=[0,-1,-2,-3]
#
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
#
# #트레이닝 횟수 1000번 ->모델생성
# #생성된 모델의 w,b,loss출력
# for i in range(1001):
#     sess.run(train,feed_dict={x:x_train,y:y_train})
#
# wv, bv, lossv= sess.run([w,b,loss],feed_dict={x:x_train,y:y_train})
# print("w값:%s b값:%s loss값:%s " %(wv,bv,lossv))
# #---------------------------------------------


# tf.set_random_seed(999)
# x=[1,2,3]
# y=[1,2,3]
# w=tf.placeholder(tf.float32)
# hf=x*w #b생략
# cost=tf.reduce_mean(tf.square(hf-y))
# sess=tf.Session()
#
# #시각화하는데 사용되는 리스트
# w_history=[]
# cost_history=[]
#
# for i in range(-40,50):
#     cw=i*0.1 # -4,-3.9,.....4.8,,4.9
#     cur_cost = sess.run(cost,feed_dict={w:cw})
#     w_history.append(cw)
#     cost_history.append(cur_cost)
#
# plt.plot(w_history,cost_history)
# plt.show()

x_data =[1,2,3]
y_data =[1,2,3]
w=tf.Variable(tf.random_normal([1]))
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
hf=x*w
cost=tf.reduce_mean(tf.square(hf-y))


############아래 4줄은 의미상 이 코딩과 같음 : tf.train.GradientDescentOptimizer(0.1).minimize(cost)
lr=0.1
gridient=tf.reduce_mean((w*x-y)*x)
descent= w-lr*gridient
update= w.assign(descent)
# ▲ update를 실행하면 w에 대한 갱신이 수행됨.

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update,feed_dict={x:x_data,y:y_data})
    # 위 문장을 수행함으로써 w가 갱신됨
    print(step,sess.run(cost,feed_dict={x:x_data,y:y_data}),sess.run(w))

