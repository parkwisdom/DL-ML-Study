import tensorflow as tf
tf.set_random_seed(777)
x1_data=[280,310,250,270,277] #모의고사 1회
x2_data=[310,300,225,267,307] #모의고사 2회
x3_data=[270,325,205,290,300] #모의고사 3회
y_data = [299,327,240,270,290] #수능점수

x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

w1=tf.Variable(tf.random_normal([1]),name='w1')
w2=tf.Variable(tf.random_normal([1]),name='w2')
w3=tf.Variable(tf.random_normal([1]),name='w3')
b=tf.Variable(tf.random_normal([1]),name='bias')
hf=x1*w1+x2*w2+x3*w3+b

cost = tf.reduce_mean(tf.square(hf-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6) #0.000001
train = optimizer.minimize(cost)

#세션 실행#
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#학습
for step in range(5001):
    cv,hv,_= sess.run([cost, hf, train],feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
    if step%10==0:
        print(step, "비용:",cv,"\n예측:",hv)





