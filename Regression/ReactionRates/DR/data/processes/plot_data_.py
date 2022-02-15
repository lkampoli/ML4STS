# compare scaling methods for mlp inputs on regression problem
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.optimizers import SGD
from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy as np

from sklearn import metrics
from sklearn.metrics import *

from sklearn.tree import DecisionTreeRegressor

# prepare dataset with input and output scalers, can be none
def get_dataset(input_scaler, output_scaler):
    # split into train and test
    n_train = 500

    X = np.genfromtxt("Temperatures.csv", delimiter=',').reshape(-1, 1)
    y = np.genfromtxt("DR_RATES-N2-N2-dis.csv", delimiter=',')

    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train,0:1], y[n_train:,0:1]

    print(X.shape)
    print(y.shape)
    print(trainX.shape)
    print(trainy.shape)

    # scale inputs
    if input_scaler is not None:
    	# fit scaler
    	input_scaler.fit(trainX)
    	# transform training dataset
    	trainX = input_scaler.transform(trainX)
    	# transform test dataset
    	testX = input_scaler.transform(testX)
    if output_scaler is not None:
    	# reshape 1d arrays to 2d arrays
    	trainy = trainy#.reshape(len(trainy), 1)
    	testy = testy#.reshape(len(trainy), 1)
    	# fit scaler on training dataset
    	output_scaler.fit(trainy)
    	# transform training dataset
    	trainy = output_scaler.transform(trainy)
    	# transform test dataset
    	testy = output_scaler.transform(testy)
    return trainX, trainy, testX, testy

# fit and evaluate mse of model on test set
def evaluate_model(trainX, trainy, testX, testy):
    # define model
    #model = Sequential()
    model = DecisionTreeRegressor()
    #model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    #model.add(Dense(1, activation='linear'))
    # compile model
    #model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
    # fit model
    #model.fit(trainX, trainy, epochs=100, verbose=0)
    model.fit(trainX, trainy)
    # evaluate the model
    #test_mse = model.evaluate(testX, testy, verbose=0)
    test_mse = mean_squared_error(testX, testy)
    return test_mse

# evaluate model multiple times with given input and output scalers
def repeated_evaluation(input_scaler, output_scaler, n_repeats=30):
	# get dataset
	trainX, trainy, testX, testy = get_dataset(input_scaler, output_scaler)
	# repeated evaluation of model
	results = list()
	for _ in range(n_repeats):
		test_mse = evaluate_model(trainX, trainy, testX, testy)
		print('>%.3f' % test_mse)
		results.append(test_mse)
	return results

# unscaled inputs
results_unscaled_inputs = repeated_evaluation(None, StandardScaler())
# normalized inputs
results_normalized_inputs = repeated_evaluation(MinMaxScaler(), StandardScaler())
# standardized inputs
results_standardized_inputs = repeated_evaluation(StandardScaler(), StandardScaler())
# summarize results
print('Unscaled: %.3f (%.3f)' % (mean(results_unscaled_inputs), std(results_unscaled_inputs)))
print('Normalized: %.3f (%.3f)' % (mean(results_normalized_inputs), std(results_normalized_inputs)))
print('Standardized: %.3f (%.3f)' % (mean(results_standardized_inputs), std(results_standardized_inputs)))
# plot results
results = [results_unscaled_inputs, results_normalized_inputs, results_standardized_inputs]
labels = ['unscaled', 'normalized', 'standardized']
pyplot.boxplot(results, labels=labels)
pyplot.show()
