import numpy as np
import time
import random

from sklearn import model_selection
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

from pandas import read_csv
from evolutionary_search import EvolutionaryAlgorithmSearchCV


class HyperparameterTuningGrid:

    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initDataset()
        self.initRegressor()
        self.initKfold()
        self.initGridParams()

    def initDataset(self):
        with open('../../data/TCs_air5.txt') as f:
            lines = (line for line in f if not line.startswith('#'))
            self.data = np.loadtxt(lines, skiprows=1)

        print(self.data.shape)
        self.X = self.data[:,0:7]         # T, P, x_N2, x_O2, x_NO, x_N, x_O
        self.y = self.data[:,7:8].ravel() # shear viscosity

    def initRegressor(self):
        self.regressor = AdaBoostRegressor(random_state=self.randomSeed)

    def initKfold(self):
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, shuffle=True,
                                           random_state=self.randomSeed)

    def initGridParams(self):
        self.gridParams = {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'learning_rate': np.logspace(-2, 0, num=10, base=10),
            'loss': ['linear', 'square', 'exponential'],
        }

    def getDefaultAccuracy(self):
        cv_results = model_selection.cross_val_score(self.regressor,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='r2')
        return cv_results.mean()

    def gridTest(self):
        print("performing grid search...")

        gridSearch = GridSearchCV(estimator=self.regressor,
                                  param_grid=self.gridParams,
                                  cv=self.kfold,
                                  scoring='r2',
                                  n_jobs=4,
                                  verbose=2)

        gridSearch.fit(self.X, self.y)
        print("best parameters: ", gridSearch.best_params_)
        print("best score: ", gridSearch.best_score_)

    def geneticGridTest(self):
        print("performing Genetic grid search...")

        gridSearch = EvolutionaryAlgorithmSearchCV(estimator=self.regressor,
                                                   params=self.gridParams,
                                                   cv=self.kfold,
                                                   scoring='r2',
                                                   verbose=True,
                                                   iid='False',
                                                   n_jobs=4,
                                                   population_size=20,
                                                   gene_mutation_prob=0.30,
                                                   tournament_size=2,
                                                   generations_number=5)
        gridSearch.fit(self.X, self.y)


def main():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create a problem instance:
    test = HyperparameterTuningGrid(RANDOM_SEED)

    print("Default Regressor Hyperparameter values:")
    print(test.regressor.get_params())
    print("score with default values = ", test.getDefaultAccuracy())

    print()
    start = time.time()
    test.gridTest()
    end = time.time()
    print("Time Elapsed = ", end - start)

    print()
    start = time.time()
    test.geneticGridTest()
    end = time.time()
    print("Time Elapsed = ", end - start)


if __name__ == "__main__":
    main()
