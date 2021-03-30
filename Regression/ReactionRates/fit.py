import time

def fit(x,y,gs):
   t0 = time.time()
   gs.fit(x, y)
   runtime = time.time() - t0
   print("Training time: %.6f s" % runtime)

   return gs
