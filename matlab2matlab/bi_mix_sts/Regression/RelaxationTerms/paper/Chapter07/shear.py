import numpy as np

from sklearn import model_selection
from sklearn import datasets

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor


class Shear1Test:
    """This class encapsulates the Shear1 regression test for feature selection
    """

    VALIDATION_SIZE = 0.20
    NOISE = 0.0

    def __init__(self, numFeatures, numSamples, randomSeed):
        """
        :param numFeatures: total number of features to be used (at least 5)
        :param numSamples: number of samples in dataset
        :param randomSeed: random seed value used for reproducible results
        """

        self.numFeatures = numFeatures
        self.numSamples = numSamples
        self.randomSeed = randomSeed

        # generate test data:
        with open('../../../../data/dataset_N2N_rhs.dat.OK') as f:
            lines = (line for line in f if not line.startswith('#'))
            data = np.loadtxt(lines, skiprows=0)

        self.X = data[:,0:56]
        self.y = data[:,56:57]

        print(data.shape)

        # divide the data to a training set and a validation set:
        self.X_train, self.X_validation, self.y_train, self.y_validation = \
                model_selection.train_test_split(self.X, self.y, test_size=self.VALIDATION_SIZE, random_state=self.randomSeed)

        scaler = MinMaxScaler()
        #scaler = StandardScaler()

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_validation = scaler.transform(self.X_validation)

        self.y_train = scaler.fit_transform(self.y_train)
        self.y_validation = scaler.transform(self.y_validation)

        print(self.X_train)
        print(self.y_train)
    
        self.regressor = GradientBoostingRegressor(random_state=self.randomSeed)
        #self.regressor = DecisionTreeRegressor(random_state=self.randomSeed)
        #self.regressor = KernelRidge()
        #self.regressor = SVR()
        #self.regressor = RandomForestRegressor(random_state=self.randomSeed)
        #self.regressor = ExtraTreesRegressor(random_state=self.randomSeed)
        #self.regressor = KNeighborsRegressor()




    def __len__(self):
        """
        :return: the total number of features
        """
        return self.numFeatures


    def getMSE(self, zeroOneList):
        """
        returns the mean squared error of the regressor, calculated for the validation set, after training
        using the features selected by the zeroOneList
        :param zeroOneList: a list of binary values corresponding the features in the dataset. A value of '1'
        represents selecting the corresponding feature, while a value of '0' means that the feature is dropped.
        :return: the mean squared error of the regressor when using the features selected by the zeroOneList
        """

        # drop the columns of the training and validation sets that correspond to the unselected features
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX_train = np.delete(self.X_train, zeroIndices, 1)
        currentX_validation = np.delete(self.X_validation, zeroIndices, 1)

        # train the regression model using th etraining set:
        self.regressor.fit(currentX_train, self.y_train.ravel())

        # calculate the regressor's output for the validation set:
        prediction = self.regressor.predict(currentX_validation)

        # return the mean square error of prediction vs actual data:
        return mean_squared_error(self.y_validation, prediction)


# testing the class:
def main():
    # create a test instance:
    test = Shear1Test(numFeatures=56, numSamples=60, randomSeed=42)


    scores = []
    # calculate MSE for 'n' first features:
    for n in range(1, len(test) + 1):
        nFirstFeatures = [1] * n + [0] * (len(test) - n)
        score = test.getMSE(nFirstFeatures)
        print("%d first features: score = %f" % (n, score))
        scores.append(score)

    # plot graph:
    sns.set_style("whitegrid")
    plt.plot([i + 1 for i in range(len(test))], scores, color='red')
    plt.xticks(np.arange(1, len(test) + 1, 1.0))
    plt.xlabel('n First Features')
    plt.ylabel('MSE')
    plt.title('MSE over Features Selected')
    plt.show()


if __name__ == "__main__":
    main()
