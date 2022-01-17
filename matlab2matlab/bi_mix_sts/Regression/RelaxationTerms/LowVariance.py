import numpy as np
from sklearn.feature_selection import VarianceThreshold

with open('../../data/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

X = data[:,0:56]   # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57]  # rhs[0:50]

print(data.shape)
print("x=",X.shape)
print("y=",y.shape)

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
print(sel.fit_transform(X))
