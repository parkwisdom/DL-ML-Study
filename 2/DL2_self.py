import tensorflow as tf
tf.set_random_seed(777)

w=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))
x=tf.placeholder(tf.float32,shape=[None])
y=tf.placeholder(tf.float32,shape=[None])

hf = x*w+b

cost = tf.reduce_mean(tf.square(hf-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(cost)


sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    sess.run(train,feed_dict={x:[1,2,3,4,5],y:[2.1,3.1,4.1,5.3,6.2]})
    if step%10==0:
        print(step, sess.run(cost,feed_dict={x:[1,2,3,4,5],y:[2.1,3.1,4.1,5.3,6.2]}))
print("-----------모델완성---------")

print("x=10? ==>", sess.run(hf,feed_dict={x:[10]}))
print("x=9.5? ==>", sess.run(hf,feed_dict={x:[9.5]}))