import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./mnist/data/", one_hot=True)

###신경망 모델 구성###
x=tf.placeholder(tf.float32, [None, 28*28])
y=tf.placeholder(tf.float32, [None, 10])

w1=tf.Variable(tf.random_normal([28*28, 256], stddev=0.01))
#입력계층과 1번째 히든계층 사이의 가중치:w1
L1=tf.nn.relu(tf.matmul(x,w1))

w2=tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2=tf.nn.relu(tf.matmul(L1,w2))

w3=tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model=tf.matmul(L2,w3)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model,labels=y))
optimizer=tf.train.AdamOptimizer(0.001).minimize(cost)
########신경망 모델 학습#########
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

batch_size=100
total_batch=int(mnist.train.num_examples/batch_size) #60만/100=>600번
for epoch in range(20):
    total_cost=0
    for i in range(total_batch):
        batch_xs, batch_ys=mnist.train.next_batch(batch_size)
        _, cv=sess.run([optimizer, cost],feed_dict={x:batch_xs, y:batch_ys})
        total_cost+=cv
    print("epoch:", "%5d" % (epoch+1),
          "avg cost:", "{:.10f}".
          format(total_cost/total_batch))
    is_correct=tf.equal(tf.argmax(model, 1), tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("정확도:", sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels}))
