import time
import sys
import numpy as np
from joblib import load

n_jobs = -1

sc_x = load(open('scaler_x_up.pkl', 'rb'))
sc_y = load(open('scaler_y_up.pkl', 'rb'))
regr = load('model_MO_kNN_VT_up.sav')

Tin    = sys.argv[1]
Xinput = np.array([[int(Tin)]])
Xinput = sc_x.transform(Xinput)

y_regr = regr.predict(Xinput)

Xinput     = sc_x.inverse_transform(Xinput)
y_regr_dim = sc_y.inverse_transform(y_regr)

np.savetxt('kVT_up.out', y_regr_dim)
