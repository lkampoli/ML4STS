import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Let's consider simply the shear viscosity file ...
with open('../../db/datasets/shear_viscosity.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=0)

X = dataset[:,0:51] # P, T, x_ci[mol+at]
y = dataset[:,51:]  # shear viscosity

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
print(sel.fit_transform(X))
