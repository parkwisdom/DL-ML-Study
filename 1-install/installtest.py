import tensorflow as tf
# hello=tf.constant('hello')
# # print(hello) #텐서에 대한 정보: Tensor("Const:0", shape=(), dtype=string)
# sess=tf.Session()
# # print(sess)
# print(sess.run(hello)) #그래프 실행 :b'hello' >>b:바이너리
# print(str(sess.run(hello),encoding='utf-8'))
#
# #########그래프 정의 부분
# a=tf.constant(5,dtype=tf.float32) # 5라는 숫자를 a라고 하자!
# b=tf.constant(10,dtype=tf.float32)
# c=tf.constant(2,dtype=tf.float32)
# d=a*b+c
# # a에 b를 곱한 후 c를 더하는 연산을 d라는 노드로 정의한다.
# ######################
#
# #########그래프 실행 부분
# sess = tf.Session()
# res=sess.run(d) #d노드르 실행한 결과를 res에 저장
# print(res)
# #######################
#
# a= tf.constant(3)
# print(a)
# # sess=tf.Session()
# # print(sess.run(a))
# #sess.close() #세션 종료 -->메모리 자원 반환
#
# with tf.Session() as sess: #sess라는 이름으로 세션객체를 만들어라.
#     #with 구문의 들여쓰기 레벨이 세션객체가 유효한 범위
#     print(sess.run(a))
#     print(a.eval())
#     #윗줄에서 세션이 종료됨.
# print(sess.run(a)) #오류
#
# a=tf.constant(5)
# b=tf.constant(3)
# c=a*b
# d=a+b
# e=c+d
# sess=tf.Session()
# print(sess.run(e))
#
# inputdata=[1,2,3]
# x=tf.placeholder(dtype=tf.float32) #실행시점에 데이터가 전달
# y=x*2 #[1,2,3]*2 =[2,4,6]
# sess= tf.Session()
# res= sess.run(y,feed_dict={x:inputdata})
# print(res)
#
# a=tf.placeholder(dtype=tf.float32)
# b=tf.placeholder("float")
# y=tf.multiply(a,b)
# z=tf.add(y,y)
# ###그래프 정의 ###
# sess=tf.Session()
# print(sess.run(y,feed_dict={a:3,b:2}))
# print(sess.run(b,feed_dict={b:5})) #a노드를 실행,
# #z노드 실행, y는 a(3)*b(2)값 =>12 출력
# print(sess.run(z,feed_dict={a:3,b:2}))


# x=tf.constant(15)
# y=tf.Variable(x+5)
# #------------▼변수 초기화------#
# sess=tf.Session()
# init=tf.global_variables_initializer()
# sess.run(init)
# #------------------------------#
# print(sess.run(y))
#
# inputdata = [1,2,3,4,5]
# x=tf.placeholder(dtype=tf.float32)
# w=tf.Variable(2,dtype=tf.float32)
# y=tf.multiply(w,x)
# sess=tf.Session()
# init=tf.global_variables_initializer()
# sess.run(init)
# res= sess.run(y,feed_dict={x:inputdata})
# print(res)
#
# x= tf.linspace(-1.0,1.0,10) # linspace : 구간을 설정할 때 쓰는 함수
#
# sess = tf.Session()
# print(sess.run(x))
# sess.close()
#
#
# a=tf.placeholder("float")
# b=tf.placeholder("float")
# y=tf.multiply(a,b)
# z= tf.add(y,y)
# with tf.Session() as sess:
#     print(sess.run(z,feed_dict={a:4,b:4,y:5})) #y값이 우선으로 적용됨
#
# x= tf.constant([[1.0,2.0,3.0]]) #1행 3열
# w= tf.constant([[2.0],[2.0],[2.0]]) #3행 1열
# y=tf.matmul(w,x) #행력의 곱셈.
# sess=tf.Session()
# res=sess.run(y)
# print(res)

# x= tf.Variable([[1.0,2.0,3.0]]) #1행 3열
# w= tf.constant([[2.0],[2.0],[2.0]]) #3행 1열
# y=tf.matmul(w,x) #행력의 곱셈.
# sess=tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# res=sess.run(y)
# print(res)


# input_data= [[1.0,2.0,3.0],[1.0,2.0,3.0],[2.0,3.0,4.0]] #3행 3열
# x=tf.placeholder(dtype=tf.float32,shape=[None,3]) #tf에서 None은 값이 정해져있지 않은 상황
# w= tf.Variable([[2.0],[2.0],[2.0]]) #3행 1열
# y=tf.matmul(x,w) #행력의 곱셈.
# sess=tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# res=sess.run(y,feed_dict={x:input_data})
# print(res)

input1=tf.constant([3.0])
input2=tf.constant([2.0])
input3=tf.constant([5.0])
inter=tf.add(input2,input3)
mul = tf.multiply(input1,inter)
with tf.Session() as sess:
    res1,res2=sess.run([mul,inter])
    mulres=sess.run(mul)
    mulinter = sess.run(inter)
print(res1)
print(res2)
print(mulinter)
print(mulres)
