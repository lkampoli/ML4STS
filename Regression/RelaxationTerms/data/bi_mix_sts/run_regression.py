import time
import sys
import numpy as np

# Serial/Parallel execution
from joblib import load
n_jobs = 1

def regressor(inp):

    # Load scalers
    sc_x = load(open('scaler_x.pkl', 'rb'))
    sc_y = load(open('scaler_y.pkl', 'rb'))

    # Load model
    regr = load('model.sav')

    # Build array of inputs for prediction
    #Xinput = np.array([inp])
    #Xinput = np.asarray(inp).reshape(1,-1) # if it is only 1 row
    Xinput = np.asarray(inp).reshape(-1,1) # if it is only 1 column

    print(Xinput)

    # Scaler input arguments
    Xinput = sc_x.transform(Xinput)

    print(Xinput)

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

    print("Tutto bene!")
    return y_regr_dim

# Test
#inp_test = np.arange(55)
#inp_test = np.arange(50)
#inp_test = np.array([6.710171944669201e+03,
#                     5.235551450558529e+02,
#                     7.094585696371519e-01,
#                     1.469012138794188e-02,
#                     1.579678337761087e-02,
#                     1.567921684329877e-02,
#                     1.381001313064924e-02,
#                     1.055991646518878e-02,
#                     7.226089307449037e-03,
#                     4.710334126328627e-03,
#                     3.031585356473885e-03,
#                     1.948303271880281e-03,
#                     1.255372066141043e-03,
#                     8.124444436789146e-04,
#                     5.285564615980966e-04,
#                     3.458202881250923e-04,
#                     2.275963220426149e-04,
#                     1.506900616479064e-04,
#                     1.003770094213713e-04,
#                     6.727098283960104e-05,
#                     4.535993752291419e-05,
#                     3.077316733221093e-05,
#                     2.100531285165122e-05,
#                     1.442595191576722e-05,
#                     9.968247629398788e-06,
#                     6.930307803137260e-06,
#                     4.847812300568716e-06,
#                     3.411919275792181e-06,
#                     2.416080724648459e-06,
#                     1.721408376118610e-06,
#                     1.234002904681323e-06,
#                     8.900374133915385e-07,
#                     6.458923811006975e-07,
#                     4.715979214337368e-07,
#                     3.464522826958772e-07,
#                     2.560794908803555e-07,
#                     1.904434300055886e-07,
#                     1.425007069556345e-07,
#                     1.072822492864603e-07,
#                     8.126406622118670e-08,
#                     6.193398948405660e-08,
#                     4.749188312896575e-08,
#                     3.664118853659767e-08,
#                     2.844327121292956e-08,
#                     2.221515563201343e-08,
#                     1.745737759572121e-08,
#                     1.380284080082785e-08,
#                     1.098039036651927e-08,
#                     8.788745914691047e-09,
#                     1.994554660071107e-01])
inp_test = 1000000
out_test = regressor(inp_test)
print(out_test)
print(out_test.shape)
print(type(out_test))
