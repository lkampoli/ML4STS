# https://medium.com/datadriveninvestor/part-ii-support-vector-machines-regression-b4d4559ba2c
# https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2
# https://github.com/tomsharp/SVR/blob/master/SVR.ipynb
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-ridge-regression-py
# https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

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

#y=np.reshape(y, (-1,1))
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#sc_x = MinMaxScaler()
#sc_y = MinMaxScaler()
#X = sc_x.fit_transform(x)
#Y = sc_y.fit_transform(y)
#y=np.reshape(y, (-1,1))
#sc_x = MinMaxScaler()
#sc_y = MinMaxScaler()
#print(sc_x.fit(x))
#X=sc_x.transform(x)
#print(sc_y.fit(y))
#Y=sc_y.transform(y)

# The data is then split into training and test data
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Just to show that linear regression is not appropriate, obviously!
# fit the regressor with x and y data
#lr = LinearRegression()
#lr.fit(x_train, y_train)

# prediction on test samples
#y_pred = lr.predict(x_test)

from sklearn.svm import SVR
 svr = SVR(kernel = 'rbf')
#svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
#svr = SVR(kernel='linear', C=100, gamma='auto')
#svr = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
svr.fit(x_train, y_train)

y_pred = svr.predict(x_test)
#y_pred=np.reshape(y_pred, (-1,1))
#y_pred = sc_y.inverse_transform(y_pred)

# Calculate metrics
# https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#plt.scatter(x[:,1], y[:], color = 'red')
plt.plot(x_test[:,1], y_pred[:], color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#plt.scatter(x[3602:4001,1], y[3602:4001], s=0.5, label='KAPPA')
#plt.scatter(x[:,1], y[:], s=0.5, label='KAPPA')
#plt.plot(x_test[:,1], y_pred[:], 'o', color='red', label='predicted', linewidth=2, markersize=2)
#plt.title(' Shear viscosity ')
#plt.ylabel(r'$\eta$ [Pa·s]')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
#plt.savefig("eta_randomforest.pdf", dpi=150, crop='false')
#plt.show()
