import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class RelaxationTermsTest:
    """This class encapsulates the RelaxationTerms regression test for feature selection
    """

    VALIDATION_SIZE = 0.20
    #NOISE = 1.0

    def __init__(self, numFeatures, numSamples, randomSeed):
        """
        :param numFeatures: total number of features to be used (at least 5)
        :param numSamples: number of samples in dataset
        :param randomSeed: random seed value used for reproducible results
        """

        self.numFeatures = numFeatures
        self.numSamples  = numSamples
        self.randomSeed  = randomSeed

        # generate test data
        with open('../../data/dataset_N2N_rhs.dat.OK') as f:
            lines = (line for line in f if not line.startswith('#'))
            data = np.loadtxt(lines, skiprows=0)

        self.X = data[0:self.numSamples, 0:self.numFeatures]
        self.y = data[0:self.numSamples, self.numFeatures+1]

        # divide the data to a training set and a validation set
        self.X_train, self.X_validation, self.y_train, self.y_validation = \
            model_selection.train_test_split(self.X, self.y, test_size=self.VALIDATION_SIZE, random_state=self.randomSeed)
 
#        sc_x = StandardScaler()
#        sc_y = StandardScaler()
#        
#        # fit scaler
#        sc_x.fit(self.X_train)
#        
#        # transform training dataset
#        self.X_train = sc_x.transform(self.X_train)
#        
#        # transform test dataset
#        self.X_validation = sc_x.transform(self.X_validation)
#        
#        # fit scaler on training dataset
#        sc_y.fit(self.y_train.reshape(-1, 1))
#        
#        # transform training dataset
#        self.y_train = sc_y.transform(self.y_train.reshape(-1, 1))
#        
#        # transform test dataset
#        self.y_validation = sc_y.transform(self.y_validation.reshape(-1, 1))

        self.regressor = GradientBoostingRegressor(random_state=self.randomSeed)

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

        # drop the columns of the training and validation sets that correspond to the
        # unselected features
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX_train = np.delete(self.X_train, zeroIndices, 1)
        currentX_validation = np.delete(self.X_validation, zeroIndices, 1)

        # train the regression model using th etraining set
        self.regressor.fit(currentX_train, self.y_train.ravel())

        # calculate the regressor's output for the validation set
        prediction = self.regressor.predict(currentX_validation)

        # return the mean square error of prediction vs actual data
        return mean_squared_error(self.y_validation, prediction)


# testing the class
def main():
    # create a test instance:
    test = RelaxationTermsTest(numFeatures=56, numSamples=1936, randomSeed=69)

    scores = []
    # calculate MSE for 'n' first features:
    for n in range(1, len(test) + 1):
        nFirstFeatures = [1] * n + [0] * (len(test) - n)
        score = test.getMSE(nFirstFeatures)
        print("%d first features: score = %f" % (n, score))
        scores.append(score)

    # plot graph
    sns.set_style("whitegrid")
    plt.plot([i + 1 for i in range(len(test))], scores, color='red')
    plt.xticks(np.arange(1, len(test) + 1, 1.0))
    plt.xlabel('n First Features')
    plt.ylabel('MSE')
    plt.title('MSE over Features Selected')
    plt.show()


if __name__ == "__main__":
    main()
