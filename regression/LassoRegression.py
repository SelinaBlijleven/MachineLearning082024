"""
LassoRegression.py (L1 Regularization)

This is an implementation of a linear regression algorithm applying L1 regularization, or LASSO
(Least Absolute Shrinkage and Selection Operator). The algorithm discourages high values for weights,
which causes some weights to be zeroed out and helps the selection process.

The L1 regularization parameter is also called lambda (which is a reserved keyword in Python). Its value
should lie between 0 and infinity. When lambda = 0 the model will act like a basic Linear Regression model
without any regularization. If lambda = infinity, all weights will be zero. These are, naturally, not the most
practical applications of this algorithm.

Note: of course many implementations already exist, this script was made for educational purposes and
can freely be used to explore the algorithm.
"""
import numpy as np
import pandas as pd


class LassoRegression:
    """ Class implementation """

    # Define our hyperparameters
    learning_rate: float                # Learning rate
    iterations: int                     # Number of iterations
    lambda_penalty: int                 # Lambda value for regularisation (range 0-inf)

    # Define training parameters
    m: int                              # Number of rows
    n: int                              # Number of features

    def __init__(self, learning_rate=0.01, iterations=1000, lambda_penalty=10) -> None:
        """
        Initialise a LassoRegressor with the given hyperparameters

        :param learning_rate:   The learning rate determines how much the weights
                                are updated during each iteration.
        :param iterations:      How many iterations to execute
        :param lambda_penalty:      The desired L1 penalty (range 0-inf, but 0 is just OLS)
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_penalty = lambda_penalty

    def fit(self, X, Y) -> 'LassoRegression':
        """
        Fit the model to the given data.

        :param X:               Given training data
        :param Y:               Given labels
        :return:                Updated model
        """
        # X contains our training data. Shape is a tuple (m, n), that tells us how many
        # rows and features we have.
        self.m, self.n = X.shape

        # We set the initial weights and the bias to zero.
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Now we can start iterating: we have our initial weights.
        # This means we can make our very first (and likely terrible prediction).
        # Afterwards, we will update the weights and try again. This process is repeated
        # for as many iterations as we specified.
        for i in range(self.iterations):
            self.update_weights()

        return self

    # Helper function to update weights in gradient descent
    def update_weights(self) -> 'LassoRegression':
        """
        Update the weights of the model.

        This function represents one iteration of learning, or one step of the gradient descent
        algorithm. In this case, regularization can be applied too.

        :return:    LassoRegressor
        """
        # Predict the y values based on the current weights
        Y_pred = self.predict(self.X)

        # Set an empty vector for the gradient, with the length of the number of features.
        dW = np.zeros(self.n)

        # Now loop over each feature to calculate the gradient of the loss function
        for j in range(self.n):
            # Calculate the gradient of the loss function relative to each weight
            if self.W[j] > 0:
                dW[j] = ((-2 * (self.X[:, j]).dot(self.Y - Y_pred) + self.lambda_penalty)
                         / self.m)
            else:
                dW[j] = ((-2 * (self.X[:, j]).dot(self.Y - Y_pred) - self.lambda_penalty)
                         / self.m)

        # Adjust the intercept (the b in y = ax + b)
        db = -2 * np.sum(self.Y - Y_pred) / self.m

        # After determining the regularized loss, we can now multiply the gradient with
        # our existing weights. The learning rate influences how large the change will be.
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X) -> float:
        """
        Helper function that uses the current weights of the model
        to predict the output y'(x), sometimes also called h(x), since it's
        a hypothetical function.

        :param X:   Data point
        :return:    Prediction
        """
        return X.dot(self.W) + self.b
