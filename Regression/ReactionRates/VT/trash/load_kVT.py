# https://joblib.readthedocs.io/en/latest/persistence.html
# https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
# https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
#
import time
import sys
import numpy as np
from joblib import load

Tin = sys.argv[1]

n_jobs = -1

#sc_x = load(open('scaler_x_up.pkl', 'rb'))
#sc_y = load(open('scaler_y_up.pkl', 'rb'))
#regr = load('model_MO_kNN_VT_up.sav')

sc_x = load(open('scaler_x_down.pkl', 'rb'))
sc_y = load(open('scaler_y_down.pkl', 'rb'))
regr = load('model_MO_kNN_VT_down.sav')

Xinput = np.array([[int(Tin)]])
Xinput = sc_x.transform(Xinput)

t0 = time.time()
y_regr = regr.predict(Xinput)
regr_predict = time.time() - t0

Xinput      = sc_x.inverse_transform(Xinput)
y_regr_dim  = sc_y.inverse_transform(y_regr)

np.savetxt('kVT_down.out', y_regr_dim)
