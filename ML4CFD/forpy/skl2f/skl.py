import numpy as np
from joblib import load

def get_scalers(scaler):
    ''' Get the names of the scalers
        and return the scalers objects

    Input: scalers name with .pkl extension
           type: string vector of dimension 2
    Output: scalers object(s)
    '''
    sc_x = load(open(scaler[0] 'rb'))
    sc_y = load(open(scaler[1], 'rb'))
    return sc_x, sc_y

def get_model(model):
    ''' Get the name of the regressor model
        and return the regressor object

    Input: regressor name with .sav extension
           type: string
    Output: regressor object
    '''
    regr = load(model)
    return regr

def transform(input, sc_x):
    tr_input = sc_x.transform(input)
    return tr_input

def transform_back(output, sc_y):
    tr_output = sc_y.inverse_transform(output)
    return tr_output

def predict(input, regr):
    output = regr.predict(input)
    return output

def run_regression(model, scaler, input, output):
    sc_x, sc_y = get_scalers(scaler)
    regr       = get_model(model)
    input      = transform(input, sc_x)
    output     = predict(input, regr)
    output     = transform_back(output, sc_y)
