#!/usr/bin/env python

import time
import sys
sys.path.insert(0, '../../Utilities/')
from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
#import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor

n_jobs = -1

dataset=np.loadtxt("./data/N2-N2-VT_down.txt")
#dataset=np.loadtxt("./data/N2-N2-VT_up.txt")

x = dataset[:,0:1]  # 0: T [K]
y = dataset[:,1:48] # k_VT rates

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

dump(sc_x, open('scaler_x_down.pkl', 'wb'))
dump(sc_y, open('scaler_y_down.pkl', 'wb'))

#dump(sc_x, open('scaler_x_up.pkl', 'wb'))
#dump(sc_y, open('scaler_y_up.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

regr = KNeighborsRegressor(n_neighbors=5,
                           algorithm='ball_tree',
                           leaf_size=100,
                           weights='distance',
                           p=1)

regr = MultiOutputRegressor(estimator=regr)

t0 = time.time()
regr.fit(x_train, y_train)
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

#print(y_regr_dim.shape)
#print(y_regr_dim)
#print(y_regr_dim[:,0:1])
#print(x_test_dim[:])
#print(y_regr_dim[11,:])

plt.scatter(x_test_dim, y_test_dim[:,5], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,5], s=2, c='purple', marker='+', label='k-NearestNeighbour, i=5')

plt.scatter(x_test_dim, y_test_dim[:,10], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='r', marker='+', label='k-NearestNeighbour, i=10')

plt.scatter(x_test_dim, y_test_dim[:,15], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,15], s=2, c='c', marker='+', label='k-NearestNeighbour, i=15')

plt.scatter(x_test_dim, y_test_dim[:,20], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,20], s=2, c='g', marker='+', label='k-NearestNeighbour, i=20')

plt.scatter(x_test_dim, y_test_dim[:,25], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,25], s=2, c='y', marker='+', label='k-NearestNeighbour, i=25')

plt.scatter(x_test_dim, y_test_dim[:,30], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,30], s=2, c='b', marker='+', label='k-NearestNeighbour, i=30')

plt.scatter(x_test_dim, y_test_dim[:,35], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,35], s=2, c='m', marker='+', label='k-NearestNeighbour, i=35')

plt.scatter(x_test_dim, y_test_dim[:,40], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,40], s=2, c='grey', marker='+', label='k-NearestNeighbour, i=40')

plt.scatter(x_test_dim, y_test_dim[:,45], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,45], s=2, c='orange', marker='+', label='k-NearestNeighbour, i=45')

#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K]')
plt.legend()
plt.tight_layout()
plt.savefig("regression_MO_kNN_VT_down.pdf", dpi=150, crop='false')
#plt.savefig("regression_MO_kNN_VT_up.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(regr, 'model_MO_kNN_VT_down.sav')
#dump(regr, 'model_MO_kNN_VT_up.sav')