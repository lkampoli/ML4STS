import numpy as np
from sklearn.feature_selection import VarianceThreshold

with open('../../data/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

X = data[:,0:56]   # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57]  # rhs[0:50]

print(data.shape)
print("Size of X=",X.shape)
print("Size of y=",y.shape)
print("X=",X)
print("y=",y)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

#sc_x = StandardScaler()
sc_x = MaxAbsScaler()
X = sc_x.fit_transform(X)
print("After scaling ...")
print("X=",X)
print("y=",y)

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

print(sel)
print("Fit_transform X =",  sel.fit_transform(X))
print("Size of X after VarianceThreshold = ", sel.fit_transform(X).shape)
