
### https://scikit-learn.org/dev/auto_examples/applications/plot_out_of_core_classification.html#example-applications-plot-out-of-core-classification-py
### https://www.programcreek.com/python/example/99242/sklearn.linear_model.SGDRegressor
### https://tomaugspurger.github.io/scalable-ml-02.html

# Import database
import numpy as np
import pandas as pd

import os
import sys
import time
import timeit
sys.path.insert(0, '../../../../Utilities/')

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
import pickle

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

n_jobs = -1
trial  = 1

import bz2
from shutil import copyfileobj

#df = pd.read_fwf("../../../Data/TCs_air5.txt")
#print(df)
#df = pd.read_fwf("../../../Data/TCs_air5.txt").to_numpy()
#print(df)
#print("df done!")

# https://stackoverflow.com/questions/6884903/how-to-pickle-several-txt-files-into-one-pickle/6897610
#file1=open('../../../Data/TCs_air5_MD.txt','r')
#obj=[file1.read()]
#pickle.dump(obj,open('result.i2','w'),2)

#df = pd.read_fwf("../../../Data/TCs_air5_MD.txt").to_numpy()
#print(df)
#print("df done!")

# https://stackoverflow.com/questions/9518705/big-file-compression-with-python
#with open('../../../Data/TCs_air5_MD.txt', 'rb') as input:
#with open('../../../Data/TCs_air5_MD_full.txt', 'rb') as input:
#    with bz2.BZ2File('../../../Data/TCs_air5_MD_full.txt.bz2', 'wb', compresslevel=9) as output:
#        copyfileobj(input, output)

# https://stackoverflow.com/questions/20428355/appending-column-to-frame-of-hdf-file-in-pandas/20428786#20428786
#store = pd.HDFStore('../../../Data/TCs_air5.h5',mode='w')
#store = pd.HDFStore('../../../Data/TCs_air5_MD.h5',mode='w')
#for chunk in pd.read_csv('../../../Data/TCs_air5.txt',chunksize=50000):
#for chunk in pd.read_fwf('../../../Data/TCs_air5.txt',chunksize=50000):
#for chunk in pd.read_fwf('../../../Data/TCs_air5_MD.txt',chunksize=100):
#         store.append('df', chunk)
#store.close()

# https://medium.com/towards-artificial-intelligence/efficient-pandas-using-chunksize-for-large-data-sets-c66bf3037f93
#data = pd.read_fwf('../../../Data/TCs_air5.txt')
#df = pd.read_fwf('../../../Data/TCs_air5.txt',chunksize=100)
#data = pd.read_fwf('../../../Data/TCs_air5_MD.txt',chunksize=100)
#data = pd.read_fwf('../../../Data/TCs_air5_MD.txt') # too big!
#df = pd.DataFrame(data)
#print(df)
#print("df done!")

# https://stackoverflow.com/questions/32358443/converting-a-dataset-into-an-hdf5-dataset
#frame = pd.read_fwf('../../../Data/TCs_air5.txt')
#type(frame)
#pd.core.frame.DataFrame
#hdf5store = pd.HDFStore('mydata.h5')
#hdf5store['frame'] = frame
##hdf5store
#list(hdf5store.items())
#hdf5store2 = pd.HDFStore('mydata.h5')
#list(hdf5store2.items())
#framecopy = hdf5store2['frame']
##framecopy
#framecopy = frame
#hdf5store2.close()

## https://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas
## create a store
#store = pd.HDFStore('mystore.h5')
#
## this is the key to your storage:
##    this maps your fields to a specific group, and defines
##    what you want to have as data_columns.
##    you might want to create a nice class wrapping this
##    (as you will want to have this map and its inversion)
#group_map = dict(
#    A = dict(fields = ['field_1','field_2'], dc = ['field_1','field_5']),
#    B = dict(fields = ['field_10'], dc = ['field_10']),
#    REPORTING_ONLY = dict(fields = ['field_1000','field_1001'], dc = []),
#)
#
#group_map_inverted = dict()
#for g, v in group_map.items():
#    group_map_inverted.update(dict([ (f,g) for f in v['fields'] ]))
#
#for f in files:
#   # read in the file, additional options may be necessary here
#   # the chunksize is not strictly necessary, you may be able to slurp each
#   # file into memory in which case just eliminate this part of the loop
#   # (you can also change chunksize if necessary)
#   for chunk in pd.read_table(f, chunksize=1000):
#       # we are going to append to each table by group
#       # we are not going to create indexes at this time
#       # but we *ARE* going to create (some) data_columns
#
#       # figure out the field groupings
#       for g, v in group_map.items():
#             # create the frame for this group
#             frame = chunk.reindex(columns = v['fields'], copy = False)
#
#             # append it
#             store.append(g, frame, index=False, data_columns = v['dc'])
#
#frame = store.select(group_that_I_want)
## you can optionally specify:
## columns = a list of the columns IN THAT GROUP (if you wanted to
##     select only say 3 out of the 20 columns in this sub-table)
## and a where clause if you want a subset of the rows
#
## do calculations on this frame
#new_frame = cool_function_on_frame(frame)
#
## to 'add columns', create a new group (you probably want to
## limit the columns in this new_group to be only NEW ones
## (e.g. so you don't overlap from the other tables)
## add this info to the group_map
#store.append(new_group, new_frame.reindex(columns = new_columns_created, copy = False), data_columns = new_columns_created)

# https://stackoverflow.com/questions/16149803/working-with-big-data-in-python-and-numpy-not-enough-ram-how-to-save-partial-r?lq=1
# https://stackoverflow.com/questions/23872942/sklearn-and-large-datasets
#a = np.memmap('test.mymemmap', dtype='float32', mode='w+', shape=(200000,1000))
## here you will see a 762MB file created in your working directory
#del a
#b = np.memmap('test.mymemmap', dtype='float32', mode='r+', shape=(200000,1000))
#b = np.memmap('test.mymemmap', dtype='float32', mode='r+', shape=(2,1000))
#b[1,5] = 123456.
#print(a[1,5])
##123456.0
#b = numpy.memmap('test.mymemmap', dtype='float32', mode='r+', shape=(2,1000), offset=150000*1000*32/8)
#b[1,2] = 999999.
#print(a[150001,2])
##999999.0

# https://towardsdatascience.com/how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff
def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and
    modify the data type to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(('Memory usage of dataframe is {:.2f}' 'MB').format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            #df[col] = df[col].astype('category')
            end_mem = df.memory_usage().sum() / 1024**2
    print(('Memory usage after optimization is: {:.2f}' 'MB').format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

#reduce_mem_usage(df)
#print(df)

#incremental_dataframe = pd.read_fwf("../../../Data/TCs_air5.txt", chunksize=100) # Number of lines to read.
#incremental_dataframe = pd.read_fwf("../../../Data/tester.txt", chunksize=46800/100) # Number of lines to read.
incremental_dataframe = pd.read_fwf("../../../Data/bigtester.txt", chunksize=11400/100) # Number of lines to read.
#idf = pd.DataFrame(incremental_dataframe).to_numpy()
#print(idf.shape)

# https://www.datacamp.com/community/tutorials/xgboost-in-python
import xgboost as xgb

# For saving regressor for next use.
lgb_estimator = None
xgb_estimator = None
estimator     = None

# First three are for incremental learning:
xgb_params = {
    'update':'refresh',
    'process_type': 'update',
    'refresh_leaf': True,
    'silent': False,
}

hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'),
                 'splitter': ('best', 'random'),
                 'max_features': ('auto', 'sqrt', 'log2'),
}]

import gc

for df in incremental_dataframe:

    data = pd.DataFrame(df).to_numpy()
    #print(data)

    #x = df.iloc[:,0:7].values
    #y = df.iloc[:,7:].values
    x = data[:,0:7]
    y = data[:,7:]

    print(x.shape)
    print(y.shape)
    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

    sc_x = StandardScaler()
    sc_y = StandardScaler()

    sc_x.fit(x_train)
    x_train = sc_x.fit_transform(x_train)
    x_test  = sc_x.fit_transform(x_test)

    sc_y.fit(y_train)
    y_train = sc_y.transform(y_train)
    y_test  = sc_y.transform(y_test)

    print(x_train)
    print(y_train)

    # https://www.programcreek.com/python/example/99828/xgboost.DMatrix
    #xgb_model = xgb.train(xgb_params,
    #                      dtrain=xgb.DMatrix(xtrain, ytrain),
    #                      evals=(xgb.DMatrix(xtest, ytest),"Valid")),
    #                      xgb_model = xgb_estimator)

    est = DecisionTreeRegressor()
    gs  = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

    t0 = time.time()
    gs.fit(x_train, y_train)
    runtime = time.time() - t0
    print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

    train_score_mse = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
    train_score_mae = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(gs.predict(x_train)))
    train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
    #train_score_me  = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
    #train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

    test_score_mse = mean_squared_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
    test_score_mae = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
    test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
    #test_score_me  = max_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
    #test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
    test_score_r2  = r2_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

    print("The model performance for testing set")
    print("--------------------------------------")
    print('MAE is {}'.format(test_score_mae))
    print('MSE is {}'.format(test_score_mse))
    print('EVS is {}'.format(test_score_evs))
    #print('ME is {}'.format(test_score_me))
    print('R2 score is {}'.format(test_score_r2))

    sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))
    #print(gs.best_params_)

    out_text = '\t'.join(['regression',
                          str(trial),
                          str(sorted_grid_params).replace('\n',','),
                          str(train_score_mse),
                          str(train_score_mae),
                          str(train_score_evs),
    #                     str(train_score_me),
    #                     str(train_score_msle),
                          str(test_score_mse),
                          str(test_score_mae),
                          str(test_score_evs),
    #                     str(test_score_me),
    #                     str(test_score_msle),
                          str(runtime)])
    print(out_text)
    sys.stdout.flush()

    best_criterion = gs.best_params_['criterion']
    best_splitter  = gs.best_params_['splitter']
    best_max_features = gs.best_params_['max_features']

    outF = open("output_TD.txt", "a")
    print('####################################', file=outF)
    print(out_text, file=outF)
    print('best_criterion = ', best_criterion, file=outF)
    print('best_splitter = ', best_splitter, file=outF)
    print('best_max_features = ', best_max_features, file=outF)
    print('MAE is {}'.format(test_score_mae), file=outF)
    print('MSE is {}'.format(test_score_mse), file=outF)
    print('EVS is {}'.format(test_score_evs), file=outF)
    #print('ME is {}'.format(test_score_me),  file=outF)
    print('R2 score is {}'.format(test_score_r2), file=outF)
    print('####################################', file=outF)
    outF.close()

    del df, x_train, y_train, x_test, y_test
    # https://www.programcreek.com/python/example/549/gc.collect
    gc.collect()

#for df in idf:
#    print(0)

# http://pandas-docs.github.io/pandas-docs-travis/user_guide/io.html#io-perf
# https://stackoverflow.com/questions/16628329/hdf5-concurrency-compression-i-o-performance
# https://stackoverflow.com/questions/38515433/pickle-dump-pandas-dataframe
def test_sql_write(df):
    if os.path.exists('test.sql'):
        os.remove('test.sql')
    sql_db = sqlite3.connect('test.sql')
    df.to_sql(name='../../../Data/test_table', con=sql_db)
    sql_db.close()

def test_sql_read():
    sql_db = sqlite3.connect('../../../Data/test.sql')
    pd.read_sql_query("select * from test_table", sql_db)
    sql_db.close()

def test_hdf_fixed_write(df):
    df.to_hdf('../../../Data/test_fixed.hdf', 'test', mode='w')

def test_hdf_fixed_read():
    pd.read_hdf('../../../Data/test_fixed.hdf', 'test')

def test_hdf_fixed_write_compress(df):
    df.to_hdf('../../../Data/test_fixed_compress.hdf', 'test', mode='w', complib='blosc')

def test_hdf_fixed_read_compress():
    pd.read_hdf('../../../Data/test_fixed_compress.hdf', 'test')

def test_hdf_table_write(df):
    df.to_hdf('../../../Data/test_table.hdf', 'test', mode='w', format='table')

def test_hdf_table_read():
    pd.read_hdf('../../../Data/test_table.hdf', 'test')

def test_hdf_table_write_compress(df):
    df.to_hdf('../../../Data/test_table_compress.hdf', 'test', mode='w',
              complib='blosc', format='table')

def test_hdf_table_read_compress():
    pd.read_hdf('../../../Data/test_table_compress.hdf', 'test')

def test_csv_write(df):
    df.to_csv('../../../Data/test.csv', mode='w')

def test_csv_read():
    pd.read_csv('../../../Data/test.csv', index_col=0)

def test_feather_write(df):
    df.to_feather('../../../Data/test.feather')

def test_feather_read():
    pd.read_feather('../../../Data/test.feather')

def test_pickle_write(df):
    df.to_pickle('../../../Data/test.pkl')

def test_pickle_read():
    pd.read_pickle('../../../Data/test.pkl')

def test_pickle_write_compress(df):
    df.to_pickle('../../../Data/test.pkl.compress', compression='xz')

def test_pickle_read_compress():
    pd.read_pickle('../../../Data/test.pkl.compress', compression='xz')

def test_parquet_write(df):
    df.to_parquet('../../../Data/test.parquet')

def test_parquet_read():
    pd.read_parquet('../../../Data/test.parquet')

# Write

#data = pd.read_fwf('../../../Data/TCs_air5.txt')
#df = pd.DataFrame(data)

start_time = time.time()
test_hdf_fixed_write(df)
print("test_hdf_fixed_write --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_fixed_write(df)
print("test_hdf_fixed_write --- %s seconds ---" % (time.time() - start_time))

#start_time = time.time()
#test_sql_write(df)
#print("test_sql_write --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_fixed_write(df)
print("test_hdf_fixed_write --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_fixed_write_compress(df)
print("test_hdf_fixed_write_compress --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_table_write(df)
print("test_hdf_table_write --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_table_write_compress(df)
print("test_hdf_table_write_compress --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_csv_write(df)
print("test_csv_write --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_feather_write(df)
print("test_feather_write --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_pickle_write(df)
print("test_pickle_write --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_pickle_write_compress(df)
print("test_pickle_write_compress --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_parquet_write(df)
print("test_parquet_write --- %s seconds ---" % (time.time() - start_time))

# Read

#start_time = time.time()
#test_sql_read()
#print("test_sql_read --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_fixed_read()
print("test_hdf_fixed_read --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_fixed_read_compress()
print("test_hdf_fixed_read_compress --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_table_read()
print("test_hdf_table_read --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_hdf_table_read_compress()
print("test_hdf_table_read_compress --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_csv_read()
print("test_csv_read --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_feather_read()
print("test_feather_read --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_pickle_read()
print("test_pickle_read --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_pickle_read_compress()
print("test_pickle_read_compress --- %s seconds ---" % (time.time() - start_time))

## The data is then split into training and test data
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)
#
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#
#sc_x.fit(x_train)
#x_train = sc_x.fit_transform(x_train)
#x_test  = sc_x.fit_transform(x_test)
#
#sc_y.fit(y_train)
#y_train = sc_y.transform(y_train)
#y_test  = sc_y.transform(y_test)
#
#dump(sc_x, open('../scaler/scaler_x_MD.pkl', 'wb'))
#dump(sc_y, open('../scaler/scaler_y_MD.pkl', 'wb'))
#
#print('Training Features Shape:', x_train.shape)
#print('Training Labels Shape:', y_train.shape)
#print('Testing Features Shape:', x_test.shape)
#print('Testing Labels Shape:', y_test.shape)
#
#regr = DecisionTreeRegressor(criterion='mse',
#                             splitter='best',
#                             max_features='auto',
#                             random_state=69)
#
#regr = MultiOutputRegressor(estimator=regr)
#
#t0 = time.time()
#with parallel_backend("dask"):
#    regr.fit(x_train, y_train)
#regr_fit = time.time() - t0
#print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)
#
#t0 = time.time()
#y_regr = regr.predict(x_test)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))
#
#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)
#y_regr_dim = sc_y.inverse_transform(y_regr)
#
#plt.scatter(x_test_dim[:,0], y_test_dim[:,0], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,0], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,0], y_test_dim[:,1], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,1], s=5, c='r', marker='d', label='k-Nearest Neighbour')
##plt.scatter(x_test_dim[:,0], y_test_dim[:,2], s=5, c='k', marker='o', label='KAPPA')
##plt.scatter(x_test_dim[:,0], y_regr_dim[:,2], s=5, c='r', marker='d', label='k-Nearest Neighbour')
##plt.title('Shear viscosity regression with kNN')
##plt.ylabel(r'$\eta$ [Pa·s]')
#plt.ylabel(' ')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
#plt.savefig("../pdf/regression_MD.pdf", dpi=150, crop='false')
#plt.show()
#
## save the model to disk
#dump(regr, '../model/model_MD.sav')
