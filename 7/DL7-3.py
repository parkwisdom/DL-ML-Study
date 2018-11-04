import tensorflow as tf
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#784개(28*28) 픽셀로 구성된 숫자 이미지 데이터
#6만개 - 트레이닝 데이터셋
#1만개 - 테스트 데이터셋 다운로드
nb_classes = 10  #0~9

x=tf.placeholder(tf.float32,[None,28*28])
y=tf.placeholder(tf.float32,[None,nb_classes])
w=tf.Variable(tf.random_normal([28*28,nb_classes]))
b=tf.Variable(tf.random_normal([nb_classes]))

hf=tf.nn.softmax(tf.matmul(x,w)+b)
cost= tf.reduce_mean(tf.reduce_sum(y*-tf.log(hf),axis=1))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
is_correct = tf.equal(tf.argmax(hf,1),tf.argmax(y,1)) #예측값:hf
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

training_epochs=15
batch_size = 100


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        #600=6만/100
        for i in range(total_batch): #600번 반복
            batch_xs,batch_ys=mnist.train.next_batch(batch_size) #100개씩 데이터 읽어옴
            c,_= sess.run([cost,optimizer],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=c/total_batch
        print('에폭:','%4d' % (epoch+1),'cost:','{:.9f}'.format(avg_cost))
    print('학습이 완료되었습니다(모델완성)')
    print("정확도: ",sess.run([accuracy],feed_dict={x:mnist.test.images,y:mnist.test.labels}))

    r=random.randint(0,mnist.test.num_examples-1)
    print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1))) #r 번째 이미지의 값
    print("Prediction:",sess.run(tf.argmax(hf,1),feed_dict={x:mnist.test.images[r:r+1]})) #r번째 값을 hf값과 비교해보기

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28))
    plt.show()