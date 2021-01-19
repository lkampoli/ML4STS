import sys
import time
import numpy as np
from joblib import load
from scipy.io import FortranFile

Tin = sys.argv[1]

n_jobs = -1

dataset = "VT_RATES-N2-N_-vt_up"

sc_x = load(open('../scaler/scaler_x_MO_'+dataset+'.pkl', 'rb'))
sc_y = load(open('../scaler/scaler_y_MO_'+dataset+'.pkl', 'rb'))

regr = load('../model/model_MO_kNN_'+dataset+'.sav')

Xinput = np.array(Tin, dtype=np.float32).reshape(1,-1)
Xinput = sc_x.transform(Xinput)

y_regr = regr.predict(Xinput)

#Xinput     = sc_x.inverse_transform(Xinput)
y_regr_dim  = sc_y.inverse_transform(y_regr)

#np.savetxt(dataset+'_MO.out', y_regr_dim)

t0 = time.time()
f = FortranFile(dataset+'_MO.unf', 'w')
f.write_record(y_regr_dim)
f.close()
d = time.time() - t0
print( "duration: %6f s." % d )
