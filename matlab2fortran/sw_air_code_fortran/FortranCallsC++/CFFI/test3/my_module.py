
from my_plugin import ffi

import time
import numpy as np
from joblib import load

# Create the dictionary mapping ctypes to np dtypes.
ctype2dtype = {}

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2**log_bytes))
        dtype = '%s%d' % (prefix[0], 2**log_bytes)
        #print( ctype )
        #print( dtype )
        ctype2dtype[ctype] = np.dtype(dtype)

# Floating point types
ctype2dtype['float'] = np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')

def asarray(ffi, ptr, shape, **kwargs):
    length = np.prod(shape)
    # Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof(ptr).item)
    #print( T )
    #print( ffi.sizeof( T ) )

    if T not in ctype2dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % T)

    a = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(T)), ctype2dtype[T]).reshape(shape, **kwargs)
    return a

def k_dr_n2(ptr):


    t0 = time.time()
    #print("Load scalers")
    sc_x = load(open('scaler_x_N2.pkl', 'rb'))
    sc_y = load(open('scaler_y_N2.pkl', 'rb'))
    runtime = time.time() - t0
    print("Scalers time %.6f s" % runtime)

    t0 = time.time()
    #print("Load model")
    regr = load('model_N2.sav')
    runtime = time.time() - t0
    print("Model time %.6f s" % runtime)

    tstart = time.time()

    t0 = time.time()
    #print("Reshape and transform input")
    Xinput = asarray(ffi, ptr, shape=(1,)).copy()
    Xinput = Xinput.reshape(1,-1)
    #print(Xinput)
    Xinput = sc_x.transform(Xinput)
    #print(Xinput)
    runtime = time.time() - t0
    print("Reshape and transformation time %.6f s" % runtime)

    t0 = time.time()
    #print("Prediction")
    y_regr = regr.predict(Xinput)
    #print(y_regr)
    runtime = time.time() - t0
    print("Prediction time %.6f s" % runtime)

    t0 = time.time()
    #print("Scale back")
    y_regr_dim = sc_y.inverse_transform(y_regr)
    #print(y_regr_dim)
    runtime = time.time() - t0
    print("Scale back time %.6f s" % runtime)

    runtime = time.time() - tstart
    print("Calling time %.6f s" % runtime)

    #FIXME: convert numpy type into ctype
    #tag = ffi.string(y_regr_dim).decode("UTF-8")
    #arr = asarray(ffi, tag, shape=(1,))
    #sarr = np.array2string(y_regr_dim, formatter={'float_kind':lambda y_regr_dim: "%.2f" % y_regr_dim})
    #sarr = np.array2string(y_regr_dim, formatter={'int':lambda y_regr_dim: hex(y_regr_dim)})
    #sarr = np.array_str(y_regr_dim)
    #arr = asarray(ffi, sarr, shape=(1,))
    #return arr

    return y_regr_dim
