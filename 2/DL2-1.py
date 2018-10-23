import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(777)
# td = tf.argmax([[5,1,4],
#                 [4,5,6]],axis=0) #2행 3열 //argmax : 같은 열끼리 비교해서 최대값의 위치 찾음
# td2 = tf.argmax([[5,1,4],
#                 [4,5,6]],axis=1) #같은 행에서 최대값의 위치 찾음
# sess= tf.Session()
# print(sess.run(td))
# print(sess.run(td2))
#
# x=np.arange(6).reshape(2,3)
# # k=tf.reduce_sum(x) #reduce_sum : 행렬의 모든 값을 더하는 함수
# # print(sess.run(k))
# print(sess.run(tf.reduce_sum(x))) #행렬 전체 요소의 합:15
# print(sess.run(tf.reduce_sum(x,axis=0))) # 열단위의 합
# print(sess.run(tf.reduce_sum(x,axis=1))) # 행단위의 합

# #y: 실제값, yhat: 예측값  --> 전달받아 cost 리턴 함수
# def costToYhat(y,yhat):
#     cost = 0
#     for i in range(len(y)):
#         cost += (y[i]-yhat[i])**2
#     # cost= y(실제)-yhat(예측)의 제곱의 합의 평균
#     return cost/len(y)
# y=[1,2,3]
# yhat=[2,4,6]
# print(costToYhat(y,yhat))

# 정규분포 :np.random.normal(size=5)
# 정규분포로부터 개수가 5개인 샘플을 추출

########임의로 1000개의 데이터셋을 만듬.
num_points =1000
vectors_set =[]
for i in range(num_points):
    x1=np.random.normal(0.0,0.55) # 평균 0,표준편차 0.55
    y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])
# print(vectors_set)
x_data = [v[0] for v in vectors_set]
y_data =[v[1] for v in vectors_set]
# print("x_data=",x_data)
# print("y_data=",y_data)
#
# plt.plot(x_data,y_data,'ro')
# plt.show()

#난수 1개를 발생시켜서 변수의 값으로 초기화하는 작업을 w라는 이름의 노드로 정의
#난수 1개 발생시켜서 w에게 주어라. w는 변수이다.
#-1에서 1사이의 난수 1개가 발생
w= tf.Variable(tf.random_uniform([1],-1.0,1.0))
b= tf.Variable(tf.zeros(1))
hf= w*x_data+b #가설함수(예측함수)

cost =tf.reduce_mean(tf.square(hf-y_data))
#tf.square :예측값-실제값의 제곱
#tf.reduce_mean:합의 평균

#a:0.1~0.01로 초기화하는게 일반적임/
#최적의 a(알파)값을 찾는 문제는 여전히 연구 분야

# W:=W-a*cost함수에 대해 미분계수(+,-)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=5)
train=optimizer.minimize(cost)
#변수 초기화
init=tf.global_variables_initializer()
############▲그래프 정의

###########▼그래프 실행
sess=tf.Session()
sess.run(init)

for step in range(8): # 트레이닝 101회 수행
    #매 트레이닝시 1000개의 데이터 cost 연산
        sess.run(train)
    # if step%10==0:
        print(sess.run(w),sess.run(b))
        print(sess.run(cost))
        print("="*50)
        plt.plot(x_data,y_data,'ro')
        plt.plot(x_data,sess.run(w)*x_data+sess.run(b))
        plt.xlabel('x')
        plt.xlim(-2,2)
        plt.ylim(0.1,0.6)
        plt.ylabel('y')
        # plt.legend()
        # plt.show()