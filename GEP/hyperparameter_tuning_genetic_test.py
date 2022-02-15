from sklearn import model_selection
from sklearn.ensemble import AdaBoostRegressor

from pandas import read_csv
import numpy as np


class HyperparameterTuningGenetic:

    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initDataset()
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, shuffle=True, random_state=self.randomSeed)

    def initDataset(self):
#        with open('../../data/TCs_air5.txt') as f:
#            lines = (line for line in f if not line.startswith('#'))
#            self.data = np.loadtxt(lines, skiprows=1)
#        self.X = self.data[:,0:7]         # T, P, x_N2, x_O2, x_NO, x_N, x_O
#        self.y = self.data[:,7:8].ravel() # shear viscosity

#        data = np.loadtxt("./data/MT/DB4T_h.dat")
#        self.X = data[:,0:10]            # press, T, TVCO2, TVO2, TVCO, x[5]
#        self.y = data[:,10:11].ravel()   # shear viscosity

        data = np.loadtxt("./data/STS/shear_viscosity.txt")
        self.X = data[:,0:51]           # press, T, TVCO2, TVO2, TVCO, x[5]
        self.y = data[:,51:52].ravel()  # shear viscosity

    # ADABoost [n_estimators, learning_rate, algorithm]:
    # "n_estimators": integer
    # "learning_rate": float
    # "loss": ['linear', 'square', 'exponential']
    def convertParams(self, params):
        n_estimators = round(params[0])                              # round to nearest integer
        learning_rate = params[1]                                    # no conversion needed
        loss = ['linear', 'square', 'exponential'][round(params[2])] # round to 0, 1 or 2 then use as index
        return n_estimators, learning_rate, loss

    def getAccuracy(self, params):
        n_estimators, learning_rate, loss = self.convertParams(params)
        self.regressor = AdaBoostRegressor(random_state=self.randomSeed,
                                            n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            loss=loss
                                           )

        cv_results = model_selection.cross_val_score(self.regressor,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='r2')
        return cv_results.mean()

    def formatParams(self, params):
        return "'n_estimators'=%3d, 'learning_rate'=%1.3f, 'loss'=%s" % (self.convertParams(params))
