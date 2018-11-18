#corpus*(망뭉치): 특정 도메인에 튿앚하는 단어들의 집합

#1. 10개의 문장 수집 -> 워드 벡터 생성
corpus=[
'boy is a young man',
'girl is a young woman',
'queen is a wise woman',
'king is a strong man',
'princess is a young queen',
'preince is a young king',
'woman is pretty',
'man is strong',
'princess is a girl will be queen',
'prince is a boy will be king']

#2. 불필요한 단어 제거
def remove_stop_words(corpus):
    #it's-> it is 등의 전처리 작업
    stop_words=['is','a','will','be']
    results=[]
    for text in corpus:
        tmp=text.split(' ')
        # print(tmp)
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    return results
    # print(results)
corpus=remove_stop_words(corpus)

words=[]
for text in corpus:
    for word in text.split(' '):
        words.append(word)
print(words)

word2int={}
for i,word in enumerate(words):
    print(i,word)
    word2int[word]=i
print(word2int)

sentences=[]
for sentence in corpus:
    sentences.append(sentence.split())
print('sentences:', sentences)

#skipgram 적용, window size:2
#[['boy','young'],['young','man']....]

WINDOW_SIZE=2
"""--word2vec--방식 
['boy','young','man']
xdata | ydata
boy[1,0,0,0]     young[0,0,1,0]
boy     young
young   boy
young   man
man     boy
man     young
"""

data=[]
for sentence in sentences:
    # print(sentence) #['boy', 'young', 'man']...
    for idx,word in enumerate(sentence):
        # print(idx,word) #0 boy  1 young....
        for neighbor in sentence[#['boy', 'young', 'man']...
                        max(idx-WINDOW_SIZE,0):
                        min(idx+WINDOW_SIZE,len(sentence))+1]:
            if neighbor!=word:
                data.append([word,neighbor ])
print("data:", data)

import pandas as pd
for text in corpus:
    print("corpus:",text)
df=pd.DataFrame(data,columns=["input","label"])
print(df.shape)

#########deep Learning
import tensorflow as tf
import numpy as np

ONE_HOT_DIM=len(words) #중복되지 않는 단어 개수

def to_one_hot_encording(data_point_index):
    one_hot_encording=np.zeros(ONE_HOT_DIM)
    one_hot_encording[data_point_index]=1
    return one_hot_encording

X=[] #input
Y=[] #target

for x,y in zip(df['input'],df['label']):
    to_one_hot_encording(word2int[x])
    X.append(to_one_hot_encording(word2int[x]))
    Y.append(to_one_hot_encording(word2int[y]))

#to_one_hot_endoding(word2int[인덱스])
# print(X)
# print(Y)

xtrain=np.asarray(X)
ytrain=np.asarray(Y)
#텐서플로우에서 사용하기 위한 다차원 배열로 변경
x=tf.placeholder(tf.float32,shape=(None,ONE_HOT_DIM))
ylabel=tf.placeholder(tf.float32, shape=(None,ONE_HOT_DIM))

EMBEDDING_DIM=2

#히든계층(워드 벡터)
w1=tf.Variable(tf.random_normal([ONE_HOT_DIM,EMBEDDING_DIM]))
b1=tf.Variable(tf.random_normal([1]))
hidden_layer=tf.add(tf.matmul(x,w1),b1)

#츨력계층(워드 벡터)
w2=tf.Variable(tf.random_normal([EMBEDDING_DIM,ONE_HOT_DIM]))
b2=tf.Variable(tf.random_normal([1]))
prediction=tf.nn.softmax(tf.add(tf.matmul(hidden_layer,w2),b2))

cost=tf.reduce_mean(-tf.reduce_sum(ylabel*tf.log(prediction), axis=1))

train=tf.train.GradientDescentOptimizer(0.05).minimize(cost)
sess=tf.Session()
init= tf.global_variables_initializer()
sess.run(init)

iteration=20000
for i in range(iteration):
    sess.run(train,feed_dict={x:xtrain,ylabel:ytrain})
    if i%3000==0:
        print("iteration"+str(i)+'cost is:',sess.run(cost,feed_dict={x:xtrain,ylabel:ytrain}))

print(sess.run(w1+b1))
vectors=sess.run(w1+b1)

df2=pd.DataFrame(vectors,columns=['x1','x2'])
df2['word']=words
df2=df2[['word','x1','x2']]
print(df2)
import matplotlib.pyplot as plt

fig,ax=plt.subplots()
for word,x1,x2 in zip(df2['word'],df2['x1'],df2['x2']):
    ax.annotate(word,(x1,x2))
padding=1.0
x_axis_min=np.amin(vectors,axis=0)[0]-padding
y_axis_min=np.amin(vectors,axis=0)[1]-padding
x_axis_max=np.amax(vectors,axis=0)[0]+padding
y_axis_max=np.amax(vectors,axis=0)[1]+padding
plt.xlim(x_axis_min,x_axis_max)
plt.ylim(y_axis_min,y_axis_max)
plt.rcParams["figure.figsize"]=(10,10)
plt.show()