import time

def predict(x_test, gs):

   t0 = time.time()
   y_regr = gs.predict(x_test)
   regr_predict = time.time() - t0
   print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

   return y_regr
