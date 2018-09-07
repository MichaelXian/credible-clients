import numpy as np
from sklearn.neural_network import MLPClassifier

class CreditModel:
    def __init__(self):
        """
        Instantiates the model object, creating class variables if needed.
        """
        n = 75
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(n, n, n), random_state=1)
        pass

    def fit(self, X_train, y_train):
        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """

        self.clf.fit(X_train, y_train)
        pass

    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """
        resultsList = self.clf.predict(X_test)
        totalNumber = 0
        totalResults = len(resultsList)
        for x in resultsList:
            totalNumber += x
        print(totalNumber/totalResults * 100, "% positive")
        return resultsList
