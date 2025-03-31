from inspect import trace

import numpy as np
####################################
x = np.array([[2],[6]])
y = np.array([[1],[0]])
X = np.array([[7,5],[3,6]])
Y = np.array([[2,4],[1,0]])
Z = np.array([[4,-1,0],[1,1,1]])

#####################################
Q1A = np.trace(np.matmul(x,x.T))
Q1B = np.trace(np.matmul(x,y.T))
Q1 = Q1A + Q1B
print("Q1", Q1)
#####################################
Q2 = np.outer(x,y)
print("Q2", Q2)
#####################################
Q3 = Y*X
print("Q3", Q3)
#####################################
Q4 = 2*(X) + 2*(Y) + np.inner(X,Y)
print("Q4", Q4)
#####################################
Q5 = np.trace(np.matmul(Y,Y.T))
print("Q5", Q5)
#####################################
Q6 = (np.trace(np.matmul(Z,Z.T))) * x
print("Q6", Q6)
#####################################
Q7 = 0
for i in range(2):
    Q7 += 2*(X[i,:])
print("Q7", Q7+1)



