#!/usr/bin/env python
# coding: utf-8

import os
import time
import sys
sys.path.insert(0, '../../../../../Utilities/')

import numpy as np
import pandas as pd
import seaborn as sns

from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import operator
import itertools

from sklearn import metrics
from sklearn.metrics import *

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.inspection import permutation_importance

from joblib import dump, load
from joblib import parallel_backend
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

import gc # garbage collection

#%time dataset=np.loadtxt("/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5_MD2.txt")
#%time dataset = pd.read_csv("/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5_MD2.txt")
#x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#y = dataset[:,7:]  # D_cidk upper triangular matrix (Dij | j=>i)
#x = df[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#y = df[:,7:]  # D_cidk upper triangular matrix (Dij | j=>i)
#dataset.head()

# https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask
#plain skipping looses heading info.  It's OK for files that don't have headings,
#or dataframes you'll be linking together, or where you make your own custom headings...
#train = pd.read_csv('../input/train.csv', skiprows=5000000, nrows=1000000, header = None, dtype=dtypes)
#but if you want to import the headings from the original file
#skip first 5mil rows, but use the first row for heading:
#train = pd.read_csv('../input/train.csv', skiprows=range(1, 5000000), nrows=1000000, dtype=dtypes)
#train.head()

# set up an empty dataframe
#df_converted = pd.DataFrame()

# we are going to work with chunks of size 100
#chunksize = 100

# in each chunk, filter for values that have 'is_attributed'==1, and merge these values into one dataframe
#for chunk in pd.read_csv('/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5_MD2.txt', chunksize=chunksize, dtype=dtypes):
#for chunk in pd.read_csv('/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5_MD2.txt', chunksize=chunksize):
#    filtered = (chunk[(np.where(chunk['is_attributed']==1, True, False))])
#    df_converted = pd.concat([df_converted, filtered], ignore_index=True, )

#df.head(10)
#import h5py
#import xarray as xr
#filename = os.path.join('data', 'accounts.*.csv')
#filename
#target = os.path.join('data', 'accounts.h5')
#target
#df_hdf = dd.read_hdf('myh4file.h5', ' ')
#df_hdf.head()

#f = h5py.File(os.path.join('.', 'myh4file.h5'), mode='r')

#delete when no longer needed
#del temp
#collect residual garbage
#gc.collect()

#make wider graphs
#sns.set(rc={'figure.figsize':(12,5)});
#plt.figure(figsize=(12,5));

n_jobs = -1

# Dataset

#def get_minibatch(doc_iter, size, pos_class=positive_class):
#    """Extract a minibatch of examples, return a tuple X_text, y.
#
#    Note: size is before excluding invalid docs with no topics assigned.
#
#    """
#    data = [('{title}\n\n{body}'.format(**doc), pos_class in doc['topics'])
#            for doc in itertools.islice(doc_iter, size)
#            if doc['topics']]
#    if not len(data):
#        return np.asarray([], dtype=int), np.asarray([], dtype=int)
#    X_text, y = zip(*data)
#    return X_text, np.asarray(y, dtype=int)
#
#
#def iter_minibatches(doc_iter, minibatch_size):
#    """Generator of minibatches."""
#    X_text, y = get_minibatch(doc_iter, minibatch_size)
#    while len(X_text):
#        yield X_text, y
#        X_text, y = get_minibatch(doc_iter, minibatch_size)
#
#
#def progress(cls_name, stats):
#    """Report progress information, return a string."""
#    duration = time.time() - stats['t0']
#    s = "%20s classifier : \t" % cls_name
#    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
#    s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
#    s += "accuracy: %(accuracy).3f " % stats
#    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
#    return s
#
## Main loop : iterate on mini-batches of examples
#for i, (X_train_text, y_train) in enumerate(minibatch_iterators):
#
#    tick = time.time()
#    X_train = vectorizer.transform(X_train_text)
#    total_vect_time += time.time() - tick
#
#    for cls_name, cls in partial_fit_classifiers.items():
#        tick = time.time()
#        # update estimator with examples in the current mini-batch
#        cls.partial_fit(X_train, y_train, classes=all_classes)
#
#        # accumulate test accuracy stats
#        cls_stats[cls_name]['total_fit_time'] += time.time() - tick
#        cls_stats[cls_name]['n_train'] += X_train.shape[0]
#        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
#        tick = time.time()
#        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
#        cls_stats[cls_name]['prediction_time'] = time.time() - tick
#        acc_history = (cls_stats[cls_name]['accuracy'],
#                       cls_stats[cls_name]['n_train'])
#        cls_stats[cls_name]['accuracy_history'].append(acc_history)
#        run_history = (cls_stats[cls_name]['accuracy'],
#                       total_vect_time + cls_stats[cls_name]['total_fit_time'])
#        cls_stats[cls_name]['runtime_history'].append(run_history)
#
#        if i % 3 == 0:
#            print(progress(cls_name, cls_stats[cls_name]))
#    if i % 3 == 0:
#        print('\n')



def scale_minibatch(x_train, x_test, y_train, y_test):
    # Data standardization
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    sc_x.fit(x_train)
    x_train = sc_x.fit_transform(x_train)
    x_test  = sc_x.fit_transform(x_test)

    sc_y.fit(y_train)
    y_train = sc_y.transform(y_train)
    y_test  = sc_y.transform(y_test)

    #dump(sc_x, open('scaler_x.pkl', 'wb'))
    #dump(sc_y, open('scaler_y.pkl', 'wb'))

    print('Training Features Shape:', x_train.shape)
    print('Training Labels Shape:',   y_train.shape)
    print('Testing Features Shape:',  x_test.shape)
    print('Testing Labels Shape:',    y_test.shape)

    return x_train, x_test, y_train, y_test


# https://adventuresindatascience.wordpress.com/2014/12/30/minibatch-learning-for-large-scale-data-using-scikit-learn/
def iter_minibatches(chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < numtrainingpoints:
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        x_chunk, y_chunk = getrows(chunkrows)
        yield x_chunk, y_chunk
        chunkstartmarker += chunksize

        # https://www.blopig.com/blog/2016/08/processing-large-files-using-python/
        # https://stackabuse.com/reading-files-with-python/
        # https://blog.richard.do/2015/07/26/read-a-file-in-chunks-in-python/

# https://stackoverflow.com/questions/519633/lazy-method-for-reading-big-file-in-python
def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        #lines = (line for line in f if not line.startswith('#'))
        data = file_object.read(chunk_size)
        if not data:
            print("break!")
            break
    print(data)
    yield data


#with open('/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5_MD2.txt') as f:
#    for piece in read_in_chunks(f):
#        process_data(piece)


def main():
    #batcherator = iter_minibatches(chunksize=1024)

    #with open('../../../../Data/TCs_air5_MD.txt') as f:
    with open('../../../../Data/tester.txt') as f:
        for datachunk in read_in_chunks(f):
            #print(datachunk)

            #x = datachunk[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
            #y = datachunk[:,7:]  # D_cidk upper triangular matrix (Dij | j=>i)

            # Delete no longer needed chunk
            del datachunk

            # Collect residual garbage
            gc.collect()

            print("GC cleaned! ")

            # Train/Test data split
            #x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

            # Scale mini-batch data
            #x_train, x_test, y_train, y_test = scale_minibatch(x_train, x_test, y_train, y_test)

            # Hyper-parameters grid
            #hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'),
            #                 'splitter': ('best', 'random'),
            #                 'max_features': ('auto', 'sqrt', 'log2'),
            #}]

            # Regressor
            #regr = DecisionTreeRegressor(random_state=69)

            # Hyper-parameters GS tuning
            #gs  = GridSearchCV(regr, cv=3, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

            # Train model
            #for x_chunk, y_chunk in batcherator:
            #    t0 = time.time()
            #    regr.partial_fit(x_chunk, y_chunk)
            #    regr_fit = time.time() - t0
            #    print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

            # Now make predictions with trained model
            #t0 = time.time()
            #y_regr = regr.predict(x_test)
            #regr_predict = time.time() - t0
            #print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

#if __name__ == '__main__':
#main()

#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)
#y_regr_dim = sc_y.inverse_transform(y_regr)

#plt.scatter(x_test_dim[:,0], y_test_dim[:,0], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,0], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,0], y_test_dim[:,1], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,1], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,0], y_test_dim[:,2], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,2], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.title('Shear viscosity regression with kNN')
#plt.ylabel(r'$\eta$ [Pa·s]')
#plt.ylabel(' ')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
#plt.savefig("regression.pdf", dpi=150, crop='false')
#plt.show()

# save the model to disk
#dump(regr, 'model.sav')

from itertools import islice

def get_chunks(file_size):
    chunk_start = 0
    chunk_size = 468 #0x20000  # 131072 bytes, default max ssl buffer size
    while chunk_start + chunk_size < file_size:
        yield(chunk_start, chunk_size)
        chunk_start += chunk_size

    final_chunk_size = file_size - chunk_start
    yield(chunk_start, final_chunk_size)



def read_file_chunked(file_path):
    with open(file_path,'r') as infile:
        while True:
            N = 468
            gen = islice(infile,N)
            datachunk = np.genfromtxt(gen, dtype=None)
            print(datachunk)
            print(datachunk.size)
            print(datachunk.shape)

            x = datachunk[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
            y = datachunk[:,7:]  # D_cidk upper triangular matrix (Dij | j=>i)

            print(x.shape)
            print(y.shape)


            print("GC cleaned!")

            # Train/Test data split
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

            # Scale mini-batch data
            x_train, x_test, y_train, y_test = scale_minibatch(x_train, x_test, y_train, y_test)

            # Hyper-parameters grid
            hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'),
                             'splitter': ('best', 'random'),
                             'max_features': ('auto', 'sqrt', 'log2'),
            }]

            # Regressor
            regr = DecisionTreeRegressor(random_state=69)

            # Hyper-parameters GS tuning
            gs  = GridSearchCV(regr, cv=3, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

            # Train model
            #for x_chunk, y_chunk in batcherator:
            t0 = time.time()
            regr.partial_fit(x_train, y_train)
            regr_fit = time.time() - t0
            print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

            #del datachunk
            #gc.collect()
            if datachunk.shape[0]<N:
                break

    #with open(file_path) as file_:
    #    file_size = os.path.getsize(file_path)
    #
    #    print('File size: {}'.format(file_size))
    #
    #    progress = 0
    #
    #    for chunk_start, chunk_size in get_chunks(file_size):
    #
    #        file_chunk = file_.read(chunk_size)

            # do something with the chunk, encrypt it, write to another file...
            #print(file_chunk)
            #datachunk = np.genfromtxt(file_chunk, dtype=None)
            #datachunk = np.loadtxt(file_chunk)
            #print(datachunk)
            #print(datachunk.shape())
            #x = datachunk[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
            #y = datachunk[:,7:]  # D_cidk upper triangular matrix (Dij | j=>i)

            #progress += len(file_chunk)
            #print('{0} of {1} bytes read ({2}%)'.format(
            #    progress, file_size, int(progress / file_size * 100))
            #)

if __name__ == '__main__':
    read_file_chunked('../../../../Data/tester.txt')
