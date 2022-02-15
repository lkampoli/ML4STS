import numpy as np
import time
import random

from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from pandas import read_csv
from evolutionary_search import EvolutionaryAlgorithmSearchCV

import hyperparameter_tuning_grid_test

def main():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create a problem instance:
    test = HyperparameterTuningGrid(RANDOM_SEED)

    print("Default Classifier Hyperparameter values:")
    print(test.classifier.get_params())
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
