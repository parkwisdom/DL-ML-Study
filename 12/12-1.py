import tensorflow as tf
import numpy as np
import random

# # 랜덤수 20000개 생성 _> fat,normal,thin->bmi.csv
# fp=open("bmi.csv",mode="w",encoding="utf-8")
# fp.write("height,weight,label\r\n") #헤더
# cnt={'thin':0,'normal':0,'fat':0}
#
# def calc_bmi(h,w):
#     bmi=w/(h/100)**2
#     if bmi < 18.5: return 'thin'
#     if bmi <25: return 'normal'
#     return 'fat'
#
# for i in range(20000):
#     h=random.randint(130,200)
#     w=random.randint(35,100)
#     label=calc_bmi(h,w)
#     cnt[label]+=1
#     fp.write("{0},{1},{2}\r\n".format(h,w,label))
# fp.close()
# print("완료,",cnt)

from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

tbl=pd.read_csv("bmi.csv",engine='python')
print(tbl)
label=tbl['label']
w=tbl["weight"]/100
h=tbl["height"]/200
wh=pd.concat([w,h],axis=1)
print(wh)

#학습용/텍스트용으로 데이터 분리

data_train,data_test,label_train,label_test=train_test_split(wh,label)

clf=svm.SVC() #svm 객체 생성
clf.fit(data_train,label_train) #모델생성

predict=clf.predict(data_test) #예측

ac_score=metrics.accuracy_score(label_test,predict)                                            #정확도
print("정답률=",ac_score)


cl_report=metrics.classification_report(label_test,predict)
print("리포트=\n",cl_report)

#6) 시각화-산점도
tbl=pd.read_csv("bmi.csv",index_col=2) #index_col=2 : fat 가리킴
#산점도 그래프 작성{자주 쓰임}
fig=plt.figure() #plt : 그림이 들어가는 곳.
ax=fig.add_subplot(1,1,1) #subplot : 윈도우가 여러 개로 나뉨 #1,1,1 : 가로로 1칸 세로로 1칸 씩 #2,1,1 : 2행 1열로 나눈 후 2번째 행에 넣어라

def scatter(lbl, color):
    b=tbl.loc[lbl]
    ax.scatter(b["weight"],b["height"],c=color, label=lbl)
scatter("fat","red")
scatter("normal","yellow")
scatter("thin","purple")
ax.legend()#범례
plt.savefig("bmi.png")

plt.show()
