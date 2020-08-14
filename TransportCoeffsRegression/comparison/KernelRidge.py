import sys
import time

import operator
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import itertools

from sklearn import kernel_ridge

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
#dataset = sys.argv[1]
dataset = np.loadtxt("../data/dataset_lite.csv", delimiter=",")
#output_file = sys.argv[2]
#trial = sys.argv[3]
trial = 1

# Read the data set into memory
#input_data = pd.read_csv(dataset, compression='gzip', sep='\t')
input_data = dataset

x = dataset[:,0:2]
y = dataset[:,2] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

TARGET_NAME = 'target'
INPUT_SEPARATOR = '\t'
n_jobs = 1

hyper_params = [{
    'kernel': ('linear', 'poly','rbf','sigmoid',),
    'alpha': (1e-4,1e-2,0.1,1,),
    'gamma': (0.01,0.1,1,10,),
}]

#sc_y = StandardScaler()
#X = StandardScaler().fit_transform(input_data.drop(TARGET_NAME, axis=1).values.astype(float))
#y = sc_y.fit_transform(input_data[TARGET_NAME].values.reshape(-1,1))

y=np.reshape(y, (-1,1))
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=None)

est=kernel_ridge.KernelRidge()

grid_clf = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=0, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
# fit model
grid_clf.fit(X_train, Y_train.ravel())
# get fit time
runtime = time.time()-t0

train_score_mse = mean_squared_error(sc_y.inverse_transform(Y_train),sc_y.inverse_transform(grid_clf.predict(X_train)))
train_score_mae = mean_absolute_error(sc_y.inverse_transform(Y_train),sc_y.inverse_transform(grid_clf.predict(X_train)))
test_score_mse = mean_squared_error(sc_y.inverse_transform(Y_test),sc_y.inverse_transform(grid_clf.predict(X_test)))
test_score_mae = mean_absolute_error(sc_y.inverse_transform(Y_test),sc_y.inverse_transform(grid_clf.predict(X_test)))

sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))

# print results
out_text = '\t'.join([dataset.split('/')[-1][:-7],
                      'kernel-ridge',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(runtime)])

print(out_text)
sys.stdout.flush()

