import sys
import numpy as np

from joblib import load

def regressor(inp):

    # Load scalers
    sc_x = load(open('scaler_x.pkl', 'rb'))
    sc_y = load(open('scaler_y.pkl', 'rb'))

    # Load model
    regr = load('model.sav')

    # Build array of inputs for prediction
    Xinput = np.asarray(inp).reshape(1,-1) # if it is only 1 row
    #Xinput = np.asarray(inp).reshape(-1,1) # if it is only 1 column

    # Scaler input arguments
    Xinput = sc_x.transform(Xinput)

    # Prediction
    y_regr = regr.predict(Xinput)

    # Inverse transformation
    y_regr_dim  = sc_y.inverse_transform(y_regr)

    return y_regr_dim
