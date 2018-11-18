import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=2)

# #행은 실제값, 열은 예측값
# array=[[5,0,0,0], #A인데, A로 예측하는 것이 5건
#        [0,10,0,0],#B인데, B로 예측하는 것이 10건
#        [0,0,15,0],#C인데, C로 예측하는 것이 15건
#        [0,0,0,5]] #D인데, D로 예측하는 것이 5건
#
# df_cm=pd.DataFrame(array, index=[i for i in "ABCD"],columns=[i for i in "ABCD"])
# print(df_cm)
# plt.figure(figsize=(10,7))
# plt.title('confusion matrix')
# sns.heatmap(df_cm,annot=True) #값의 크기에 따라 색상을 다르게 설정 : 열지도(heatmap)
# plt.show()


# array=[[9,1,0,0],
#        [1,15,3,1],
#        [5,0,24,1],
#        [0,4,1,15]]
# df_cm=pd.DataFrame(array, index=[i for i in "ABCD"],columns=[i for i in "ABCD"])
# print(df_cm)
# plt.figure(figsize=(10,7))
# plt.title('confusion matrix')
# sns.heatmap(df_cm,annot=True)
# plt.show()

##randomforest를 이용하여 mnist분류기 제작
from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mnist =datasets.load_digits()
features, labels =mnist.data,mnist.target
print(np.shape((features))) #데이터
print(np.shape((labels))) #답

#만든 모델에 대하 검증하기
def cross_validation(classifier, features, labels):
    cv_scores=[]
    for i in range(10):
        scores=cross_val_score(classifier,features,labels,cv=10,scoring='accuracy') #검증평가점수
        cv_scores.append(scores.mean())
    return cv_scores

dt_cv_scores=cross_validation(tree.DecisionTreeClassifier(),features,labels)
rf_cv_scores=cross_validation(RandomForestClassifier(),features,labels)

cv_list=[
    ['random forest',rf_cv_scores],
    ['decision tree',dt_cv_scores],
]

df=pd.DataFrame.from_items(cv_list)
df.plot()
plt.show()

print(np.mean(dt_cv_scores))
print(np.mean(rf_cv_scores))

