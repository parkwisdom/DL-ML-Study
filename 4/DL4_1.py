import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# print(math.e)

##########sigmoid 함수의 형태 알아보기
# def sigmoid(z):
#     return 1/(1+math.e**-z)
# print(sigmoid(-10)) #0에 가까운..
# print(sigmoid(0)) #0.5
# print(sigmoid(10)) #무한대에 가까운
#
# for i in range(-100,100):
#     z=i/10 #z=wx+b
#     s=sigmoid(z)
#     plt.plot(z,s,'ro')
# # plt.show()
#
# def A():
#     return 'A'
# def B():
#     return 'B'
#
# y=1
# print(y*A()+(1-y)*B()) #cost 함수


def without_normalization():
    data = [[828.659973, 833.450012, 908100, 828.349976, 831.659973],
            [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
            [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
            [816, 820.958984, 1008100, 815.48999, 819.23999],
            [819.359985, 823, 1188100, 818.469971, 818.97998],
            [819, 823, 1198100, 816, 820.450012],
            [811.700012, 815.25, 1098100, 809.780029, 813.669983],
            [809.51001, 816.659973, 1398100, 804.539978, 809.559998]]
    ##데이터 단위의 차이가 커서 발산이 일어남.==> 정규화 작업 필요!!!!!!!!111
    print(np.shape(data))
    data=np.transpose(data)
    x=data[:-1].transpose().astype(np.float32)  #(8,4)
    y=data[-1:].transpose().astype(np.float32)  #(8,1)
    print(x.shape)
    print(y.shape)

    #############▼그래프 정의
    w=tf.Variable(tf.random_uniform([4,1],-1,1))
    b=tf.Variable(tf.random_uniform([1],-1,1))
    hf= tf.matmul(x,w)+b #hf=x*w ==> [8,1]=[8,4]*[4,1]

    cost = tf.reduce_mean(tf.square(hf-y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000000001)
    train = optimizer.minimize(cost)
    #----------------------------------

    #세션 생성, 변수 초기화
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    #훈련
    for i in range(201):
        sess.run(train)
        if i%5==0:
            print(i, sess.run(cost))



data = [[828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998]]


# def min_max_scaler(data): #정규화 함수
#     print(np.min(data))  #804.539978 -->data 전체의 최소값
#     print(np.max(data))  #1828100.0 -->data 전체의 최대값
#     print("-"*50)
#     print(np.min(data,axis=0))  #axis=0 :열방향으로 비교
#     print(np.max(data,axis=0))  #[8.28659973e+02 8.33450012e+02 1.82810000e+06 8.28349976e+02 8.31659973e+02]
#     min=np.min(data,axis=0)
#     max=np.max(data,axis=0)
#
#     #data를 정규화한 결과 리턴!
#     return (data-min)/(max-min)
# print(min_max_scaler(data))

def min_max_scaler_by_row(data): #정규화 함수
    rowmin=np.min(data,axis=1)
    rowmax=np.max(data,axis=1)
    # print(np.min(data,axis=1))  #axis=0 :행방향으로 비교
    # print(np.max(data,axis=1))

    return(np.transpose(data)-rowmin)/(rowmax-rowmin)
print(min_max_scaler_by_row(data))
