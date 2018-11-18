#PCA
import pandas as pd
df = pd.DataFrame(columns=['calory', 'breakfast', 'lunch', 'dinner', 'exercise', 'body_shape'])
#5차원 데이터
df.loc[0] = [1200, 1, 0, 0, 2, 'Skinny']
df.loc[1] = [2800, 1, 1, 1, 1, 'Normal']
df.loc[2] = [3500, 2, 2, 1, 0, 'Fat']
df.loc[3] = [1400, 0, 1, 0, 3, 'Skinny']
df.loc[4] = [5000, 2, 2, 2, 0, 'Fat']
df.loc[5] = [1300, 0, 0, 1, 2, 'Skinny']
df.loc[6] = [3000, 1, 0, 1, 1, 'Normal']
df.loc[7] = [4000, 2, 2, 2, 0, 'Fat']
df.loc[8] = [2600, 0, 2, 0, 0, 'Normal']
df.loc[9] = [3000, 1, 2, 1, 1, 'Fat']
print(df)

X=df[['calory', 'breakfast', 'lunch', 'dinner', 'exercise']]
print(X)
Y=df[['body_shape']]
print(Y)
#스케일링
from sklearn.preprocessing import StandardScaler
x_std=StandardScaler().fit_transform(X)
print(x_std)

import numpy as np

#공분산( x=(10,5),xt=(5,10))
print(x_std.shape)
features=x_std.T
print(features.shape)
covariance_matrix=np.cov(features) #각 피처가 행단위로 구성되도록 transpose
print('-'*50)
print(covariance_matrix)

#고유벡터, 고유값
eig_vals, eig_vecs=np.linalg.eig(covariance_matrix)
print("고유벡터를 출력합니다:\n%s" %eig_vecs)
print('고유값을 출력합니다:\n%s' %eig_vals)
print(eig_vals[0]/sum(eig_vals)) #0.7318321731427544 => 73%의 특성을 담을 수 있다.

print(x_std.shape)
print(eig_vecs.T[0].shape)

projected_X=x_std.dot(eig_vecs.T[0])#5차원 데이터를 1번째 고유벡터로 fitting(5차원->1차원)
print(projected_X)

res=pd.DataFrame(projected_X,columns=['PC1'])
res['y-axis']=0.0
res['label']=Y
print(res)

import matplotlib.pyplot as plt
import seaborn as sns
sns.lmplot('PC1','y-axis',data=res, fit_reg=False,
           scatter_kws={'s':50}, #maker 크기
           hue='label') #색상
plt.title('PCA result')
plt.show()