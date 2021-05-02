import h5py
import numpy as np

from os.path import dirname, join as pjoin
import scipy.io as sio

#f = h5py.File('./data_species.mat','r')
#om_e = np.array(f['OMEGA'])

#data_dir = pjoin(dirname(sio.__file__), './')
#mat_fname = pjoin(data_dir, 'data_species.mat')

# Load the .mat file contents.
#mat_contents = sio.loadmat(mat_fname)

mat_contents = sio.loadmat('data_species.mat',struct_as_record=False)

print(sorted(mat_contents.keys()))

om_e = mat_contents['OMEGA']
print(om_e)

import scipy.io
mat = scipy.io.loadmat('data_species.mat',struct_as_record=False)
#mat.dtype
print(sorted(mat.keys()))
#print(mat.shape)

om_e   = mat['OMEGA']
om_x_e = mat['OMEGA']
print(om_e)
print(om_x_e)
#print(om_e[0])
#print(om_e[1])
