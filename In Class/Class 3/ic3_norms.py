from numpy import genfromtxt
import numpy as np

A = genfromtxt('test_A.csv', delimiter=',')
B = genfromtxt('test_B.csv', delimiter=',')
C = genfromtxt('test_C.csv', delimiter=',')
x = genfromtxt('test_x.csv', delimiter=',')
y = genfromtxt('test_y.csv', delimiter=',')
z = genfromtxt('test_z.csv', delimiter=',')
print("A\n",A)
print("B\n",B)
print("C\n",C)
print("x\n",x)
print("y\n",y)
print("z\n",z, "\n")

print("\n",1)
print(A.T @ A)

print("\n",2)
print(A*B)

print("\n",3)
print(np.trace(np.outer(x,y)))

print("\n",4)
print(A@B.T)

print("\n",5)
print(A@B.T @ C @ x)

print("\n",6)
print((C@x + C@y))

print("\n",7)
print(np.count_nonzero(x))
print(np.linalg.norm(x, ord=1))
print(np.linalg.norm(x, ord=2))
print(np.linalg.norm(x, ord=np.inf))

print("\n",8)
print(np.count_nonzero(x))
print(np.linalg.norm(x, ord=1))
print(np.linalg.norm(x, ord=2))
print(np.linalg.norm(x, ord=np.inf))

print("\n", 9)
print(np.count_nonzero(x))
print(np.linalg.norm(x, ord=1))
print(np.linalg.norm(x, ord=2))
print(np.linalg.norm(x, ord=np.inf))




