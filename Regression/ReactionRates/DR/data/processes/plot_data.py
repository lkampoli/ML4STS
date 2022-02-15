import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeRegressor


# prepare dataset with input and output scalers, can be none
def get_dataset(input_scaler, output_scaler):

    X = np.genfromtxt("Temperatures.csv", delimiter=',').reshape(-1, 1)
    y = np.genfromtxt("DR_RATES-N2-N2-dis.csv", delimiter=',')

    X = X.reshape(-1, 1)

    # scale inputs
    if input_scaler is not None:
        # fit scaler
        input_scaler.fit(X)
        # transform training dataset
        X = input_scaler.transform(X)
    if output_scaler is not None:
        # reshape 1d arrays to 2d arrays
        y = y#.reshape(len(y), 1)
        # fit scaler on training dataset
        output_scaler.fit(y)
        # transform training dataset
        y = output_scaler.transform(y)
    return X, y


ss  = StandardScaler()
mm  = MinMaxScaler()
mas = MaxAbsScaler()
rs  = RobustScaler(quantile_range=(25, 75))
pt1 = PowerTransformer(method="yeo-johnson")
pt1 = PowerTransformer(method="box-cox")
qt1 = QuantileTransformer(output_distribution="uniform")
qt2 = QuantileTransformer(output_distribution="normal")
n   = Normalizer()

X_none, y_none  = get_dataset(None, None)
X_ss,   y_ss    = get_dataset(ss, ss)
X_mm,   y_mm    = get_dataset(mm, mm)
X_mas,  y_mas   = get_dataset(mas, mas)
X_rs,   y_rs    = get_dataset(rs, rs)
X_pt1,  y_pt1   = get_dataset(pt1, pt1)
X_pt2,  y_pt2   = get_dataset(pt2, pt2)
X_qt1,  y_qt1   = get_dataset(qt1, qt1)
X_qt2,  y_qt2   = get_dataset(qt2, qt2)
X_n,    y_n     = get_dataset(n, n)

fig, axs = plt.subplots(3, 3)

axs[0, 0].plot(X_none, y_none[:,0])
axs[0, 0].set_title("none")
#axs[0, 0].set_yscale('log')

axs[1, 0].plot(X_ss, y_ss[:,0])
axs[1, 0].set_title("ss")
#axs[1, 0].set_yscale('log')

axs[2, 0].plot(X_mm, y_mm[:,0])
axs[2, 0].set_title("mm")
#axs[2, 0].set_yscale('log')

axs[0, 1].plot(X_mas, y_mas[:,0])
axs[0, 1].set_title("mas")
#axs[0, 1].set_yscale('log')

axs[1, 1].plot(X_rs, y_rs[:,0])
axs[1, 1].set_title("rs")
#axs[1, 1].set_yscale('log')

axs[2, 1].plot(X_pt1, y_pt1[:,0])
axs[2, 1].set_title("pt1")
#axs[2, 1].set_yscale('log')

axs[0, 2].plot(X_pt2, y_pt2[:,0])
axs[0, 2].set_title("pt2")
#axs[0, 2].set_yscale('log')

axs[1, 2].plot(X_qt1, y_qt1[:,0])
axs[1, 2].set_title("qt1")
#axs[1, 2].set_yscale('log')

axs[2, 2].plot(X_qt2, y_qt2[:,0])
axs[2, 2].set_title("qt2")
#axs[2, 2].set_yscale('log')

for ax in axs.flat:
    ax.set(xlabel='T [K]', ylabel='k [?]')

plt.savefig('scaled_rates.pdf')
fig.tight_layout()
plt.show()
