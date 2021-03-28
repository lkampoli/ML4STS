from scipy.io import FortranFile
import numpy as np
f = FortranFile('test.unf', 'w')
arr = np.array([1,2,3,4,5], dtype=np.float32)
print(type(arr))
print(arr.shape)
print(arr)
f.write_record(arr)
#f.write_record(np.linspace(0,1,20).reshape((5,4)).T)
f.close()
