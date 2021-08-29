import numpy as np
def multi_label(x):
    return (x > 0)


a=np.array([[0.05,0.03,-0.02],[0.07,-0.07,0.03]])
b=multi_label(a)
print(b)