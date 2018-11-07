# CNN-MNIST

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20181105)
tf.set_random_seed(20181105)

#load data
mnist=input_data.read_data_sets('tmp/data/',one_hot=True)

num_filters1=32
x=tf.placeholder(tf.float32,[None,784]) #28*28*1
x_image=tf.reshape(x,[-1,28,28,1]) #28*28*1 행렬을 무한개(-1) 연산의 편의를 위해 2차원 행렬 할것임.
# 784(1차원 벡터)->28변환행렬
W_conv1=tf.Variable(tf.random_normal([5,5,1,num_filters1])) #3번째 인자가 1인 이유가 흑백이기 때문에
#W_conv1의 차원(shape):[5,5,1,32]가 됨.

#필터에 입력이미지 적용
h_conv1=tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding="SAME")
#strides : 이미지 왼쪽상단부터 한칸씩 이동하면서 적용 파라미터로 정한 수대로 이동설정[1,가로,상하,1]
#padding : 필터를 적용하면 출출된 특징 핼렬은 원 이미지보다 크기가 계속 작아짐->특징이 유실..문제가 발생될 수 있음->그래서 패딩함.

#필터적용이 끝나면 -> 활성화 함수 적용(CNN-ReLu함수)
#바이어스에 필터 적용한 값에 relu적용?!
b_conv1=tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_cutoff=tf.nn.relu(h_conv1+b_conv1)

#pooling: max pooling 을 적용
#max pooling 적용 의미? 추출된 특징 모두를 사용해서 특징을 판단하지 않고, 일부 특징만 가지고 판단

h_pool1=tf.nn.max_pool(h_conv1_cutoff,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#ksize : 풀링시 필터(커널)의 크기 2*2크기로 묶어서 풀링함.
#strides: 오른쪽으로 2칸 아래쪽으로 2칸씩 이동


#행렬의 차원 변환



#두번째 컴볼루셔널 계층
num_filters2=64 #필터의 개수
W_conv2=tf.Variable(tf.random_normal([5,5,num_filters1,num_filters2])) #[5,5,32,64]
#필터 크기: 5*5, 입력되는 값:32개, 32개가 들어가서 총 64개의 필터가 적용됨.
h_conv2=tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding="SAME")
b_conv2=tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_cutoff=tf.nn.relu(h_conv2+b_conv2)
h_pool2=tf.nn.max_pool(h_conv2_cutoff,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#28*28-> 첫번째풀링(2*2)->14*14->두번째 풀링(2*2)->7*7 ::: 결국 7*7크기의 행렬이 64개가 나오게 됨.

#풀리 커넥티드 계층
#64개의 입력으로 부터 ====>10개의 숫자로 분류
#두번째 컨볼루셔널 계층으로 특징을 뽑아냈으면, 이 특징으로 0~9까지의 숫자를 판별하기 위한 fully connected layer구성

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*num_filters2]) #입력된 64개의 7*7 행렬을 1차원 행렬로 변환

num_units1=7*7*num_filters2
num_units2=1024

w2=tf.Variable(tf.random_normal([num_units1,num_units2]))
b2=tf.Variable(tf.constant(0.1,shape=[num_units2]))
hidden2=tf.nn.relu(tf.matmul(h_pool2_flat,w2)+b2)
keep_prob=tf.placeholder(tf.float32) #drop out을 위해 줄 keep_prod 직접설정하기 위함.
hidden2_drop=tf.nn.dropout(hidden2,keep_prob) #drop out 해주기

w0=tf.Variable(tf.zeros([num_units2,10]))
b0=tf.Variable(tf.zeros([10]))
k=tf.matmul(hidden2_drop,w0)+b0

p=tf.nn.softmax(k) #k에 대한 softmax.
#########################################모델 정의 끝##

####비용 함수 정의
#cost :크로스 엔트로피 함수 적용
t=tf.placeholder(tf.float32,[None,10])

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=k,labels=t)) #k:예측, t:실제값
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction=tf.equal(tf.argmax(p,1),tf.argmax(t,1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()


####학습 수행
i=0
for _ in range(1000):
    i+=1
    batch_xs,batch_ts= mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch_xs,t:batch_ts,keep_prob:0.5}) #keep_prob 50%로 설정
    if i%500==0:
        loss_vals,acc_vals=[],[] #학습비용, 정확도
        for c in range(4): #test,lables의 1만개
            start=int(len(mnist.test.labels)/4*c) #0
            end=int(len(mnist.test.labels)/4*(c+1)) #2500
            loss_val,acc_val=sess.run([loss,accuracy],feed_dict={x:mnist.test.images[start:end],
                                                                 t:mnist.test.labels[start:end],
                                                                 keep_prob:1.0})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            loss_val = np.sum(loss_vals)
            acc_val = np.mean(acc_vals)

            print("Step:%d, Loss:%f, Accuracy:%f" %(i,loss_val,acc_val))

saver.save(sess,'cnn_session')
sess.close()