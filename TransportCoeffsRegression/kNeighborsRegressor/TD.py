
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

# Import database
dataset=np.loadtxt("../data/dataset_TD.csv", delimiter=",")
x=dataset[:,0:3]
y=dataset[:,3] # 0: X, 1: T, 2: I, 3: TD (thermal diffusion)

# 2D Plot
#plt.scatter(dataset[:,1], dataset[:,3], s=0.5)
#plt.title('Thermal diffusion')
#plt.xlabel('T [K]')
#plt.ylabel(r'$D_T$ $[m^2/s]$')
#plt.tight_layout()
#plt.savefig("thermal_diffusion_2d.pdf", dpi=150)
#plt.show()

# 3D Plot
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(dataset[:,0], dataset[:,1], dataset[:,3], s=0.5)
#ax.set_xlabel('molar fraction', rotation=150)
#ax.set_ylabel('T [K]')
# disable auto rotation
#ax.zaxis.set_rotate_label(False)
#ax.set_zlabel(r'$D_T$ $[m^2/s]$', rotation = 0, labelpad=13)
#ax.dist = 14
#plt.savefig("thermal_diffusion_3d.pdf", dpi=150)
#plt.show()

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

neigh = KNeighborsRegressor(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                            metric_params=None, n_jobs=None)

neigh.fit(x_train, y_train)

# prediction on test samples
y_pred = neigh.predict(x_test)

# Calculate metrics
# https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(x[:,1], y[:], s=0.5, label='KAPPA')
plt.plot(x_test[:,1], y_pred[:], 'o', color='red', label='predicted', linewidth=2, markersize=2)
plt.title('Thermal diffusion')
plt.ylabel(r'$D_T$ $[m^2/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("TD_kneighbors.pdf", dpi=150, crop='false')
plt.show()
