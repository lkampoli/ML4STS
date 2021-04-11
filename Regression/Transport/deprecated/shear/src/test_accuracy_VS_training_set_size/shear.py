# study of training set size for an mlp on the circles problem
from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential
from numpy import mean
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from keras.callbacks import TensorBoard

from keras.utils.vis_utils import plot_model
from keras.models import load_model

from keras.models import model_from_json

from keras_sequential_ascii import keras2ascii

from keras.optimizers import SGD, Adam, RMSprop, Adagrad

import numpy as np

# create train and test datasets
def create_dataset(n_train, n_test=100000, noise=0.1):
	# generate samples
	n_samples = n_train + n_test
	X, y = make_circles(n_samples=n_samples, noise=noise, random_state=1)
	# split into train and test, first n for test
	trainX, testX = X[n_test:, :], X[:n_test, :]
	trainy, testy = y[n_test:], y[:n_test]
	# return samples
	return trainX, trainy, testX, testy

# evaluate an mlp model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	model = Sequential()
	model.add(Dense(70, input_dim=7, activation='relu'))
	model.add(Dense(35, activation='relu'))
	model.add(Dense(1, activation='linear'))
	opt = keras.optimizers.Adam(learning_rate=0.01, decay=1e-3/100)
	model.compile(loss='mse', metrics=['mse', 'mae', 'mape', 'msle'], optimizer=opt)
#       model.add(Dense(1, activation='sigmoid'))
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=500, verbose=2)
	# evaluate the model
#	_, test_acc = model.evaluate(testX, testy, verbose=0)
	_, test_acc = model.predict(testX)
	return test_acc

# repeated evaluation of mlp model with dataset of a given size
def evaluate_size(n_train, n_repeats=5):
	# create dataset
	with open('../../../../Data/TCs_air5.txt') as f:
		lines = (line for line in f if not line.startswith('#'))
		dataset = np.loadtxt(lines, skiprows=1)
	print(dataset.shape)
	x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
	y = dataset[:,7:8] # shear
	#trainX, trainy, testX, testy = create_dataset(n_train)
	trainX, testX, trainy, testy = train_test_split(x, y, train_size=n_train, test_size=0.2, random_state=69)
	# repeat evaluation of model with dataset
	scores = list()
	for _ in range(n_repeats):
		# evaluate model for size
		score = evaluate_model(trainX, trainy, testX, testy)
		scores.append(score)
	return scores

# define dataset sizes to evaluate
#sizes = [100, 1000, 5000, 10000]
sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # train set size
score_sets, means = list(), list()
for n_train in sizes:
	# repeated evaluate model with training set size
	scores = evaluate_size(n_train)
	score_sets.append(scores)
	# summarize score for size
	mean_score = mean(scores)
	means.append(mean_score)
	print('Train Size=%d, Test Accuracy %.3f' % (n_train, mean_score*100))
# summarize relationship of train size to test accuracy
pyplot.plot(sizes, means, marker='o')
pyplot.show()
# plot distributions of test accuracy for train size
pyplot.boxplot(score_sets, labels=sizes)
pyplot.show()
