import sys
#import time
import numpy as np
from joblib import load

# Input: temperature, T[K]
Tin = sys.argv[1]

# Load scalers
sc_x = load(open('scaler_x_NO.pkl', 'rb'))
sc_y = load(open('scaler_y_NO.pkl', 'rb'))

# Load model
regr = load('model_NO.sav')
#regr = tf.keras.models.load_model('model.sav')

# Reshape and transform input
Xinput = np.array(Tin, dtype=np.float32).reshape(1,-1)
Xinput = sc_x.transform(Xinput)

# Prediction
#t0 = time.time()
y_regr = regr.predict(Xinput)
#d = time.time() - t0
#print( "Predict: %6f s." % d )

#Xinput    = sc_x.inverse_transform(Xinput)
y_regr_dim = sc_y.inverse_transform(y_regr)

# Save results
#t0 = time.time()
np.savetxt('result_NO.out', y_regr_dim)
#d = time.time() - t0
#print( "Save: %6f s." % d )

# fastest
#t0 = time.time(); open(dataset+'_MO.bin', "wb").write(y_regr_dim);  d = time.time() - t0; print( "duration: %6f s." % d )
