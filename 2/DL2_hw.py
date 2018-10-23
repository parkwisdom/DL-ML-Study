import numpy as np
import tensorflow as tf
# xy=np.loadtxt('xy.txt',skiprows=1)
# print(xy)
# print(xy.shape)
# print(type(xy))
#
# xxy=np.loadtxt('xxy.csv',delimiter=',')
# print(xxy)
#
# xxy2=np.loadtxt('xxy.csv',delimiter=',',unpack=True)
# print(xxy2)

cars=np.loadtxt('cars.csv',delimiter=',')
# print(cars)
# print(cars.shape[0])

# 각 요소별로 분리하기
xx=[]
yy=[]
# for i in range(len(cars)):
#     item=cars[i]
#     xx.append(item[0])
#     yy.append(item[1])
# print(xx)
# print('----------')
# print(yy)

for row in cars:
    xx.append(row[0])
    yy.append(row[1])
# print(xx)
# print('----------')
# print(yy)

for speed,distance in cars:
    xx.append(speed)
    yy.append(distance)
# print(xx)
# print('----------')
# print(yy)
######################

cars=np.loadtxt('cars.csv',delimiter=',',unpack=True)
print(cars.shape)
print(cars)
x=cars[0] #스피드
y=cars[1] #제동거리

cars=cars.transpose()
print(cars.shape)

