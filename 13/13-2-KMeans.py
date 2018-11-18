# import math
# import pandas
# with open('data/nba_2013.csv','r') as csvflie:
#     nba=pandas.read_csv(csvflie)
# # print(nba)
# # print(nba.columns)
#
# distance_columns=[ 'age', 'g', 'gs', 'mp', 'fg', 'fga',
#        'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft',
#        'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf','pts']
#
# selected_player=nba[nba['player']=="LeBron James"].iloc[0] #여기서 선정한 선수에 대해 가까운것을 찾자
#
# def euclidean_distance(row):
#     inner_value=0
#     for k in distance_columns:
#         inner_value+=(selected_player[k]-row[k])**2
#     return math.sqrt(inner_value)
#
# LeBron_distance=nba.apply(euclidean_distance,axis=1)
# # print(LeBron_distance)
# # print('-'*50)
# nba_numeric=nba[distance_columns]
# # print(nba_numeric)
# # print('-'*50)
# nba_normalized=(nba_numeric-nba_numeric.mean())/nba_numeric.std() #정규화
# # print(nba_normalized)
#
# from scipy.spatial import distance
#
# nba_normalized.fillna(0,inplace=True) #inplace=True : 기존 객체에 저장된 값을 바꾼다.
#
# LeBron_normalized=nba_normalized[nba['player']=='LeBron James']
#
# euclidean_distances=nba_normalized.apply(lambda  row: distance.euclidean(row,LeBron_normalized),axis=1) #만들어진 distanc에 해당되는 값
# print(euclidean_distances)
#
# distance_frame=pandas.DataFrame(data={"dist":euclidean_distances,"idx":euclidean_distances.index})
# distance_frame.sort_values("dist",inplace=True)
# print(distance_frame.iloc[1]['idx']) #가장 가까운 선수 추출
# # distance_frame.iloc[:2,2]
#
# # 선수 이름출력
# second_smallest =distance_frame.iloc[1]['idx']
# most_similar_to_Lebron=nba.loc[int(second_smallest)]['player']
# print("가장 비슷한 성적의 선수:",most_similar_to_Lebron)
#
# # # nba.apply()



from sklearn import datasets
import pandas as pd
iris=datasets.load_iris()
# print(iris)
labels=pd.DataFrame(iris.target)
# print(labels)
labels.columns=['labels']
# print(labels)

data=pd.DataFrame(iris.data)
data.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
# print(data)
# print('-'*40)
# print(labels)
# print('-'*40)

data=pd.concat([data,labels],axis=1)
print(data)
print('-'*40)
feature=data[['Sepal_Length','Sepal_width']]
# print(feature)
# print(feature.head(10))
# print(feature.tail(10))


#만들어진 kmeans알고리즘 사용
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

model=KMeans(n_clusters=5,algorithm='auto')
model.fit(feature)

predict=pd.DataFrame(model.predict(feature))
predict.columns=['predict']
print(predict)
r=pd.concat([feature,predict],axis=1)
print(r)
plt.scatter(r['Sepal_Length'],r['Sepal_width'],c=r['predict'],alpha=0.5)
# plt.show()
centers=pd.DataFrame(model.cluster_centers_,columns=['Sepal_Length','Sepal_width'])
print(centers)

centers_x=centers['Sepal_Length']
centers_y=centers['Sepal_width']
plt.scatter(centers_x,centers_y,s=50,marker='D',c='r') #x좌표, y좌표, 크기, 마커, 색상
# plt.show()

from sklearn.pipeline import make_pipeline
#make_pipline 매서드는 scaler와 kmeans를 순차적으로 실행시키는 기능을 수행
from sklearn.preprocessing import StandardScaler #표준화해주는 매서드

model=KMeans(n_clusters=3) #KMeans를 수행할 수 있는 객체를 생성. 클러스터를 3개로 만드는 모델 변수를 정의

scaler=StandardScaler()
pipline=make_pipeline(scaler,model)
pipline.fit(feature)

predict=pd.DataFrame(pipline.predict(feature))
ks=range(1,10)
inertias=[]
for k in ks:
    model=KMeans(n_clusters=k)
    model.fit(feature)
    inertias.append(model.inertia_)
#inertia_:inertia value 를 이용해서 적정 수준의 클러스터 개수 파악 ,
# 클러스터 마다의 거리와 클러스터 내부데이터 거리 값을 합함 -> 작을수록 응집이 잘된거
plt.plot(ks,inertias,'-0')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
ct=pd.crosstab(data['labels'],r['predict'])
print(ct)

make_pipeline()
