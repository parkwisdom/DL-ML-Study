import tensorflow as tf
import numpy as np

data = np.loadtxt('data.csv',delimiter=',',dtype='float32', encoding='utf-8',unpack=True)
print(data)
xdata=np.transpose(data[0:2])
ydata=np.transpose(data[2:])
print(xdata)
print(ydata)

#신경망 모델 생성
global_step=tf.Variable(0,trainable=False,name='global_step')
# global_step: 모델 저장과정에서 사용될 변수 / 초기값:0

x=tf.placeholder(tf.float32)#None,2
y=tf.placeholder(tf.float32)#None,3

w1=tf.Variable(tf.random_uniform([2,10],-1.,1.))
L1=tf.nn.relu(tf.matmul(x, w1))

w2=tf.Variable(tf.random_uniform([10,20],-1.,1.))
L2=tf.nn.relu(tf.matmul(L1, w2))

w3=tf.Variable(tf.random_uniform([20,3],-1.,1.))
model=tf.matmul(L2, w3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=model))
optimizer= tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(cost,global_step=global_step)

#신경망 모델 생성 (학습)
sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())
#global_variables:변수를 가져오는 함수->파일 저장

ckpt=tf.train.get_checkpoint_state('./model') #기존에 학습된 모델이 저장된 폴더가(/model) 있는지 확인하는 함수

#모델이 저장된 폴더가 존재하면서 모델이 만들어져 있다면 (ckpt이면서 모델이 만들어져 있다면)...
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

for step in range(2):
    sess.run(train,feed_dict={x:xdata,y:ydata})
    print('step: %d' % sess.run(global_step),'cost: %.3f' % sess.run(cost,feed_dict={x:xdata,y:ydata}))

saver.save(sess, './model/dnn.ckpt',global_step=global_step)