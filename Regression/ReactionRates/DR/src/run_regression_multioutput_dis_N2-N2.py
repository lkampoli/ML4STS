import sys
#import time
import numpy as np
from joblib import load
from scipy.io import FortranFile

Tin = sys.argv[1]

n_jobs = -1

dataset = "DR_RATES-N2-N2-dis"

sc_x = load(open('../scaler/scaler_x_MO_'+dataset+'.pkl', 'rb'))
sc_y = load(open('../scaler/scaler_y_MO_'+dataset+'.pkl', 'rb'))

regr = load('../model/model_MO_kNN_'+dataset+'.sav')

Xinput = np.array(Tin, dtype=np.float32).reshape(1,-1)
Xinput = sc_x.transform(Xinput)

y_regr = regr.predict(Xinput)

#Xinput     = sc_x.inverse_transform(Xinput)
y_regr_dim  = sc_y.inverse_transform(y_regr)
#print(y_regr_dim.dtype) #float64

#t0 = time.time()
np.savetxt(dataset+'_MO.out', y_regr_dim)
#d = time.time() - t0
#print( "duration: %6f s." % d )

# fastest
#t0 = time.time(); open(dataset+'_MO.bin', "wb").write(y_regr_dim);  d = time.time() - t0; print( "duration: %6f s." % d )

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFile.html
#t0 = time.time()
f = FortranFile(dataset+'_MO.unf', 'w')
f.write_record(y_regr_dim)
f.close()
#d = time.time() - t0
#print( "duration: %6f s." % d )

#t1 = time.time(); FortranFile(dataset+'_MO.unf', 'w').write_record(y_regr_dim); d1 = time.time() - t1; print( "duration: %6f s." % d1 )

#f = FortranFile(dataset+'_MO.unf', 'r')
#print(f.read_reals(float))
