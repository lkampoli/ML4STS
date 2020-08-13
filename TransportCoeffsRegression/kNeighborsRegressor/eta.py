# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.neighbors import NearestCentroid

# Import database
dataset=np.loadtxt("../data/dataset_lite.csv", delimiter=",")
x=dataset[:,0:2]
y=dataset[:,2] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

# Plot dataset
#plt.scatter(x[:,1], dataset[:,2], s=0.5)
#plt.title('Shear viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\eta$')
#plt.show()

#plt.scatter(x[:,1], dataset[:,3], s=0.5)
#plt.title('Bulk viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\zeta$')
#plt.show()

#plt.scatter(x[:,1], dataset[:,4], s=0.5)
#plt.title('Thermal conductivity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\lambda$')
#plt.show()

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

neigh = KNeighborsRegressor(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                            metric_params=None, n_jobs=None)

#neigh = KNeighborsRegressor(n_neighbors=1, weights='uniform')
#neigh = KNeighborsRegressor(n_neighbors=1, weights='distance')
#neigh = RadiusNeighborsRegressor(radius=0.4)
neigh.fit(x_train, y_train)

# prediction on test samples
y_pred = neigh.predict(x_test)

print(neigh.kneighbors(x_test, return_distance=True))

# Calculate metrics
# https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#plt.scatter(x[3602:4001,1], y[3602:4001], s=0.5, label='KAPPA')
plt.scatter(x[:,1], y[:], s=0.5, label='KAPPA')
plt.plot(x_test[:,1], y_pred[:], 'o', color='red', label='predicted', linewidth=2, markersize=2)
plt.title(' Shear viscosity ')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("eta_kneighbors.pdf", dpi=150, crop='false')
plt.show()
