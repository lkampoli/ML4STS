import sys
import time
import numpy as np
from joblib import load
from scipy.io import FortranFile

# Input: X
Xin = sys.argv[1]

# Load scalers
sc_x = load(open('scaler_x.pkl', 'rb'))
sc_y = load(open('scaler_y.pkl', 'rb'))

# Load model
regr = load('model.sav')

# Reshape and transform input
Xinput = np.array(Xin, dtype=np.float32).reshape(1,-1)
Xinput = sc_x.transform(Xinput)

# Prediction
#t0 = time.time()
y_regr = regr.predict(Xinput)
#d = time.time() - t0
#print( "Predict: %6f s." % d )

#Xinput    = sc_x.inverse_transform(Xinput)
y_regr_dim = sc_y.inverse_transform(y_regr)

# 10 times slower than .bin
# Save results
#t0 = time.time()
#np.savetxt('result_XY.out', y_regr_dim)
#d = time.time() - t0
#print( "duration: %6f s." % d )
#print( "Solution: ", y_regr_dim )
#print( "Size: ", y_regr_dim.size )

# 2 times slower than .bin
#t0 = time.time()
#f = FortranFile('result_XY.unf', 'w')
#f.write_record(y_regr_dim)
#f.close()
#d = time.time() - t0
#print( "duration: %6f s." % d )

# fastest
t0 = time.time(); open('result_XY.bin', "wb").write(y_regr_dim);  #d = time.time() - t0; print( "duration: %6f s." % d )
