import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x>0,dtype=np.int)
# x=np.arange(-5.0,5.0,0.1)
# y=step_function(x)
# # print(y)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

def relu(x):
    return np.maximum(0,x)
# x=np.arange(-5.0,5.0,0.1)
# y=relu(x)
# plt.plot(x,y)
# plt.ylim(-0.1,5.5)
# plt.show()

A=np.array([[1,2],[3,4],[5,6]])
print(A.shape)

#행렬의 내적
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
print(np.dot(A,B))

A=np.array([[1,2,3],[4,5,6]])
B=np.array([[1,2],[3,4],[5,6]])
print(np.dot(A,B))

A=np.array([[1,2],[3,4],[5,6]])
B=np.array([7,8])
print(np.dot(A,B))
#---------------------

#신경망 내적 구하기
x=np.array([1,2])
w=np.array([[1,3,5],[2,4,6]])

def sigmoid(x):
    return 1/(1+np.exp(-x))

X=np.array(([1.0,0.5]))
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
A1=np.dot(X,W1)+B1
print(A1)
print("-"*50)
Z1=sigmoid(A1)
print(Z1)

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)
print(A2,Z2)

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
print(A3)
