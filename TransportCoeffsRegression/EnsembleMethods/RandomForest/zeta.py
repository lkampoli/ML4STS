# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# https://www.geeksforgeeks.org/random-forest-regression-in-python/
# https://github.com/afrozchakure/Internity-Summer-Internship-Work/tree/master/Blogs/Random_Forest_Regression
# https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454f
# https://stackoverflow.com/questions/8961586/do-i-need-to-normalize-or-scale-data-for-randomforest-r-package
# https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb
# https://towardsdatascience.com/multivariate-time-series-forecasting-using-random-forest-2372f3ecbad1
# https://github.com/WillKoehrsen/Data-Analysis/tree/master/random_forest_explained
# https://github.com/WillKoehrsen/Data-Analysis/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%201.ipynb
# https://github.com/WillKoehrsen/Data-Analysis/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb
# https://stackoverflow.com/questions/32664717/got-continuous-is-not-supported-error-in-randomforestregressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.score

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

# Import database
dataset=np.loadtxt("../data/dataset_lite.csv", delimiter=",")
x=dataset[:,0:2]
y=dataset[:,3] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

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

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
#regressor = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None,
#                                  min_samples_split = 2, min_samples_leaf = 1)

# fit the regressor with x and y data
regressor.fit(x_train, y_train)

# prediction on test samples
y_pred = regressor.predict(x_test)

# Calculate metrics
# https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#plt.scatter(x[3602:4001,1], y[3602:4001], s=0.5, label='KAPPA')
plt.scatter(x[:,1], y[:], s=0.5, label='KAPPA')
plt.plot(x_test[:,1], y_pred[:], 'o', color='red', label='predicted', linewidth=2, markersize=2)
plt.title(' Bulk viscosity for molar fraction = 0.9 ')
plt.ylabel(r'$\zeta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("zeta_randomforest.pdf", dpi=150, crop='false')
plt.show()
