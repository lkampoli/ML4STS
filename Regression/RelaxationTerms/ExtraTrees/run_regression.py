import time
import sys
import numpy as np

# Parallel execution
from joblib import load
n_jobs = -1

# Load scalers
sc_x = load(open('scaler_x.pkl', 'rb'))
sc_y = load(open('scaler_y.pkl', 'rb'))

# Load model
regr = load('model.sav')

# Build array of inputs for prediction
Xinput = np.array([[int(Tin)]])
Xinput = sc_x.transform(Xinput)

# Prediction
t0 = time.time()
y_regr = regr.predict(Xinput)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (Xinput.shape[0], regr_predict))

# Inverse transformation
Xinput      = sc_x.inverse_transform(Xinput)
y_regr_dim  = sc_y.inverse_transform(y_regr)

# Save to file (not necessary in Matlab)
#np.savetxt(dataset+'.out', y_regr_dim)
