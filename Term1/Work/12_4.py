import numpy as np

w = [[-0.5,0.2,0.1], [0.7,-0.8,0.2]]
x = [0.2,0.5,0.6]
b = [0.1,0.2]

y = np.dot(w,x) + b
print("y:=", y)