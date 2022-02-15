import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

X = np.genfromtxt("DR/data/processes/Temperatures.csv", delimiter=',').reshape(-1, 1)

y_DRd = np.genfromtxt("DR/data/processes/DR_RATES-N2-N2-dis.csv", delimiter=',')
y_DRr = np.genfromtxt("DR/data/processes/DR_RATES-N2-N2-rec.csv", delimiter=',')

y_VT = np.genfromtxt("VT/data/processes/VT_RATES-N2-N2-vt_down.csv", delimiter=',')

y_VV = np.genfromtxt("VV/data/processes/VV_DOWN_N2_RATES-0-.1.csv", delimiter=',')

y_ZR = np.genfromtxt("ZR/data/processes/ZR_N2_RATES-N2i_+O_=_NO0_+N.csv", delimiter=',')

X = X.reshape(-1, 1)

for i in range(0,y_DRd.shape[1]-1):
    plt.scatter(X,y_DRd[:,i], s=2, c='k', marker='x', label='DRd')
    plt.scatter(X,y_DRr[:,i], s=2, c='r', marker='x', label='DRr')
    plt.scatter(X,y_VT[:,i],  s=2, c='b', marker='x', label='VT')
    plt.scatter(X,y_VV[:,i],  s=2, c='g', marker='x', label='VV')
    plt.scatter(X,y_ZR[:,i],  s=2, c='m', marker='x', label='ZR')
    plt.yscale('log')
plt.show()
plt.close()

#fig1 = plt.figure(1)
for i in range(0,y_DRd.shape[1]-1):
    plt.scatter(X,y_DRd[:,i], s=2, c='k', marker='x', label='DRd')
    plt.yscale('log')
#plt.legend()
#plt.tight_layout()
plt.show()
plt.close()

for i in range(0,y_DRr.shape[1]-1):
    plt.scatter(X,y_DRr[:,i], s=2, c='r', marker='x', label='DRr')
    plt.yscale('log')
#plt.legend()
#plt.tight_layout()
plt.show()
plt.close()

for i in range(0,y_VT.shape[1]-1):
    plt.scatter(X,y_VT[:,i], s=2, c='b', marker='x', label='VT')
    plt.yscale('log')
#plt.legend()
#plt.tight_layout()
plt.show()
plt.close()

for i in range(0,y_VV.shape[1]-1):
    plt.scatter(X,y_VV[:,i], s=2, c='g', marker='x', label='VV')
    plt.yscale('log')
#plt.legend()
#plt.tight_layout()
plt.show()
plt.close()

for i in range(0,y_ZR.shape[1]-1):
    plt.scatter(X,y_ZR[:,i], s=2, c='m', marker='x', label='ZR')
    plt.yscale('log')
#plt.legend()
#plt.tight_layout()
plt.show()
plt.close()

#plt.scatter(X,y_VT[:,0], s=2, c='r', marker='o', label='VT')
#plt.scatter(X,y_VV[:,0], s=2, c='g', marker='o', label='VV')
#plt.scatter(X,y_ZR[:,0], s=2, c='b', marker='o', label='ZR')

#plt.xlabel('T [K]')
#plt.legend()
#plt.tight_layout()
#plt.savefig('processes.pdf')
#plt.show()
#plt.close()
