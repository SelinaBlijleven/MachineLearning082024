"""
LassoRegressionTest.py

A unit test for the LassoRegression model, to make sure it works as intended.
This file is just an extra quality assurance measure, not course content.
"""
import unittest
import numpy as np

# Import our own implementation of LassoRegression
from LassoRegression import LassoRegression


class TestLassoRegression(unittest.TestCase):

    def setUp(self) -> None:
        """ Set up our test environment with some generated data. """
        # Set a reproducible seed for our 'random' synthetic data.
        np.random.seed(42)

        # Create 100 samples with 5 features (rand range 0-1).
        self.X = np.random.rand(100, 5)
        self.true_weights = np.array([1.5, -2.0, 0.0, 3.0, 0.0])
        self.y = self.X.dot(self.true_weights) + np.random.randn(100) * 0.5  # Linear combination plus noise

        # Initialize the Lasso Regression Model
        self.model = LassoRegression(learning_rate=0.01, iterations=1000, lambda_penalty=10)

    def test_fit(self):
        """ Test if the model can fit the data without errors """
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.W, "Weights should not be None after fitting")
        self.assertEqual(len(self.model.W), self.X.shape[1], "Weights length should match number of features")

    def test_weight_regularization(self):
        """ Test if Lasso regularization leads to some weights being zero (or close to zero) """
        self.model.fit(self.X, self.y)
        # Count weights close to zero
        zeroed_weights = np.sum(np.abs(self.model.W) < 1e-2)
        self.assertGreater(zeroed_weights, 0, "No weights zeroed out despite Lasso regularization")

    def test_model_performance(self):
        """ Check if the model can produce a reasonable error """

        # Fit and predict
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)

        # Calculate and check the MSE
        mse = np.mean((self.y - y_pred) ** 2)
        self.assertLess(mse, 1.0, "Mean Squared Error should be reasonably low")


if __name__ == "__main__":
    unittest.main()
