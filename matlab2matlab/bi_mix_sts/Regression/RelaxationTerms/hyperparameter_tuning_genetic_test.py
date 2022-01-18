from sklearn import model_selection
from sklearn.ensemble import AdaBoostRegressor

from pandas import read_csv
import numpy as np


class HyperparameterTuningGenetic:

    NUM_FOLDS = 10

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initDataset()
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, shuffle=True, random_state=self.randomSeed)

    def initDataset(self):
        with open('../../data/dataset_N2N_rhs.dat.OK') as f:
            lines = (line for line in f if not line.startswith('#'))
            data = np.loadtxt(lines, skiprows=0)

        self.X = data[:,0:56]
        self.y = data[:,56:57]

    # ADABoost [n_estimators, learning_rate, loss]:
    # "n_estimators": integer
    # "learning_rate": float
    # "loss": {'linear', 'square', 'exponential'}
    def convertParams(self, params):
        n_estimators = round(params[0])  # round to nearest integer
        learning_rate = params[1]        # no conversion needed
        loss = ['linear', 'square', 'exponential'][round(params[2])]  # round to 0 or 1, then use as index
        return n_estimators, learning_rate, loss

    def getScore(self, params):
        n_estimators, learning_rate, loss = self.convertParams(params)
        self.regressor = AdaBoostRegressor(random_state=self.randomSeed,
                                           n_estimators=n_estimators,
                                           learning_rate=learning_rate,
                                           loss=loss
                                          )

        cv_results = model_selection.cross_val_score(self.regressor,
                                                     self.X,
                                                     self.y.ravel(),
                                                     cv=self.kfold,
                                                     scoring='r2')
        return cv_results.mean()

    def formatParams(self, params):
        return "'n_estimators'=%3d, 'learning_rate'=%1.3f, 'loss'=%s" % (self.convertParams(params))
