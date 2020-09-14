#import time
import sys
import numpy as np
from joblib import load

Tin = sys.argv[1]
Lev = sys.argv[2]

n_jobs = -1

dataset   = "DR_RATES-N2-N2-dis"

sc_x = load(open('scaler_x_'+dataset+'_'+Lev+'.pkl', 'rb'))
sc_y = load(open('scaler_y_'+dataset+'_'+Lev+'.pkl', 'rb'))

regr = load('model_kNN_'+dataset+'_'+Lev+'.sav')

Xinput = np.array([[int(Tin)]])
Xinput = sc_x.transform(Xinput)

#t0 = time.time()
y_regr = regr.predict(Xinput)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (Xinput.shape[0], regr_predict))

#Xinput      = sc_x.inverse_transform(Xinput)
y_regr_dim  = sc_y.inverse_transform(y_regr)

np.savetxt(dataset+'.out', y_regr_dim)
