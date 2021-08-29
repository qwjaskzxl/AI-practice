import numpy as np

A = np.ones([80, 60, 60])

B = np.random.randn(60,1)

C = np.dot(A,B)
print(A)
print(B)
print(C.shape)