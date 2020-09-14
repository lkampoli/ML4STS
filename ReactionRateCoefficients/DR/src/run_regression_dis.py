import sys
import numpy as np
from joblib import load
#from scipy.io import FortranFile

Tin = sys.argv[1]
Lev = sys.argv[2]

n_jobs = -1

#dataset   = "DR_RATES-N2-N2-dis"
dataset   = "DR_RATES-N2-N-dis"

sc_x = load(open('scaler_x_'+dataset+'_'+Lev+'.pkl', 'rb'))
sc_y = load(open('scaler_y_'+dataset+'_'+Lev+'.pkl', 'rb'))

regr = load('model_kNN_'+dataset+'_'+Lev+'.sav')

Xinput = np.array(Tin, dtype=np.float32).reshape(1,-1)
Xinput = sc_x.transform(Xinput)

y_regr = regr.predict(Xinput)

#Xinput     = sc_x.inverse_transform(Xinput)
y_regr_dim  = sc_y.inverse_transform(y_regr)

np.savetxt(dataset+'.out', y_regr_dim)

#file = open(dataset+'.bin', "wb")
#file.write(y_regr_dim)
#file.close()

#f = FortranFile(dataset+'.unf', 'w')
#f.write_record(y_regr_dim)
#f.close()
