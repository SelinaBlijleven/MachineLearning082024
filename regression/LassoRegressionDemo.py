"""
LassoRegressionDemo.py

Demonstrate the Lasso regularized regression in action!
"""
# For dummy data and numerical adventures.
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from LassoRegression import LassoRegression


class LassoRegressionDemo:

    # Number of samples to use
    N_SAMPLES = 100

    # Noise factor (will become the standard deviation of the noise)
    # The more noise we have, the higher the regularization will have to be
    NOISE_FACTOR = 0.5

    # Set some weights for the real model, with some features being zeroed out
    TRUE_WEIGHTS = np.array([1.5, -2.0, 0.0, 3.0, 0.0])

    # Different configurations for Lasso (L1) Regression
    # Format: [learning_time, iterations, lambda]
    MODEL_PROTOTYPES = [
        {"learning_rate": 0.01, "iterations": 1000, "lambda_penalty": 0},
        {"learning_rate": 0.01, "iterations": 1000, "lambda_penalty": 1},
        {"learning_rate": 0.01, "iterations": 1000, "lambda_penalty": 10},
        {"learning_rate": 0.01, "iterations": 1000, "lambda_penalty": 50},
        {"learning_rate": 0.01, "iterations": 1000, "lambda_penalty": 1000000},
    ]

    def main(self):
        X_train, X_test, y_train, y_test = self.generate_data()

        for config in self.MODEL_PROTOTYPES:
            # Initialize the Lasso Regression Model, entering the
            # model parameters in the given order.
            lasso = LassoRegression(**config)

            # Fit the Model
            lasso.fit(X_train, y_train)

            # Make Predictions on the test set
            y_pred = lasso.predict(X_test)

            # Evaluate the model
            print(f"\nModel: \n {config}")
            mse = mean_squared_error(y_test, y_pred)
            print(y_test[:20], y_pred[:20])
            print(f"Mean Squared Error on the test set: {mse:.4f}")

            # 7. Output the Learned Weights
            print("Learned Weights:", lasso.W)
            print("Learned Intercept:", lasso.b)

            # 8. Compare the Learned Weights with the True Weights
            print("True Weights:", self.TRUE_WEIGHTS)

    def generate_data(self):
        """
        Generate some synthetic data using real pre-set weights.

        :return:    Dummy feature and target data (train and test)
        """
        # Set the random seed for consistency
        np.random.seed(42)

        # Create some random values between 0-1 for X.
        X = np.random.rand(self.N_SAMPLES, len(self.TRUE_WEIGHTS))

        # Determine the true labels and add noise (mu=0, sigma=1 * NOISE_FACTOR)
        y = X.dot(self.TRUE_WEIGHTS) + np.random.standard_normal(self.N_SAMPLES) * self.NOISE_FACTOR

        # Make the train-test split
        return train_test_split(X, y, test_size=0.3)


if __name__ == "__main__":
    LassoRegressionDemo().main()
