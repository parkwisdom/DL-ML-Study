import tensorflow as tf

sess = tf.Session()
myqueue =tf.train.string_input_producer(["d:/dl/data/q_1.txt","d:/dl/data/q_2.txt","d:/dl/data/q_3.txt"],shuffle=False)

reader = tf.TextLineReader()
key,value = reader.read(myqueue)
record_default=[[-1],[999]]
sp, dist = tf.decode_csv(value, record_defaults=record_default)

x_batch,y_batch = tf.train.batch([sp,dist],batch_size=4) #한번에 4개씩 데이터를 읽기

# print(sess.run(key))
# print(sess.run(value))
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(100):
    x,y=sess.run([x_batch,y_batch])
    #x와 y를 이용하여 모델링
    print(x,y)
coord.request_stop()
coord.join(threads=threads)
