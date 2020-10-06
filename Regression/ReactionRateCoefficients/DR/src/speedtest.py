import time
import sys
import numpy as np
from joblib import load
#from scipy.io import FortranFile

Tin = sys.argv[1]
#Lev = sys.argv[2]

n_jobs = -1

dataset = "DR_RATES-N2-N2-dis"

sc_x = load(open('../scaler/scaler_x_MO_'+dataset+'.pkl', 'rb'))
sc_y = load(open('../scaler/scaler_y_MO_'+dataset+'.pkl', 'rb'))

regr = load('../model/model_MO_kNN_'+dataset+'.sav')

Xinput = np.array(Tin, dtype=np.float32).reshape(1,-1)
Xinput = sc_x.transform(Xinput)

y_regr = regr.predict(Xinput)

Xinput     = sc_x.inverse_transform(Xinput)
y_regr_dim = sc_y.inverse_transform(y_regr)

t0 = time.time()
np.savetxt(dataset+'_MO.out', y_regr_dim)
d = time.time() - t0
print( "duration: %6f s." % d )

t0 = time.time(); open("bla.txt", "wb").write(y_regr_dim);  d = time.time() - t0; print( "duration: %6f s." % d )
