import numpy as np
from sklearn.model_selection import train_test_split

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,20,30,40,50,60,70,80,90,100]
z = [11,21,31,41,51,61,71,81,91,101]

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.5, random_state=42)

print(x_train, x_test)
print(y_train, y_test)
print(z_train, z_test)
