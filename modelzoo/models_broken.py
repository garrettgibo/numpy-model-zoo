"""Collection of unfinished models"""
import numpy as np
from typing import Tuple
import tqdm


def z_standardize(X_inp):
    """
        Z-score Standardization.
        Standardize the feature matrix, and store the standarize rule.

        Parameter:
        X_inp: Input feature matrix.

        Return:
        Standardized feature matrix.
    """

    toreturn = X_inp.copy()
    for i in range(X_inp.shape[1]):
        # Find the standard deviation of the feature
        std = np.std(X_inp[:, i])
        # Find the mean value of the feature
        mean = np.mean(X_inp[:, i])
        temp = []
        for j in np.array(X_inp[:, i]):
            temp += [(j - mean) / std]
        toreturn[:, i] = temp
    return toreturn


def sigmoid(x):
    """Sigmoid Function

    :returns: transformed x.
    """
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, alpha=0.001, epochs=1000, early_stop=0):
        """Logisitic Regression Initialization"""
        self.alpha = alpha
        self.epochs = epochs
        self.early_stop = early_stop
        pass

    def fit(self, X_train, y_train):
        """Save the datasets in our model, and do normalization to y_train

        :param X_train: Matrix or 2-D array. Input feature matrix.
        :param Y_train: Matrix or 2-D array. Input target value.
        """

        self.X = X_train
        self.y = y_train

        count = 0
        uni = np.unique(y_train)
        for y in y_train:
            if y == min(uni):
                self.y[count] = -1
            else:
                self.y[count] = 1
            count += 1

        n, m = X_train.shape
        self.theta = np.zeros(m)
        self.b = 0

    def gradient(
        self, X_inp: np.ndarray, y_inp: np.ndarray, theta: np.array, b: int
    ) -> Tuple[np.array, int]:
        """Calculate the grandient of Weight and Bias, given sigmoid_yhat,
        true label, and data.

        :param X_inp: Matrix or 2-D array. Input feature matrix.
        :param y_inp: Matrix or 2-D array. Input target value.
        :param theta: Matrix or 1-D array. Weight matrix.
        :param b: int. Bias.

        :returns grad_theta: gradient with respect to theta
        :returns grad_b: gradient with respect to b

        NOTE: There are several ways of implementing the gradient. We are
        merely providing you one way of doing it. Feel free to change the code
        and implement the way you want.
        """
        grad_b = self.b
        grad_theta = theta

        """
            TODO: 3. Update grad_b and grad_theta using the Sigmoid function
        """
        # for (xi, yi) in zip(X_inp, y_inp):
        #     grad_b += (self.y - yi)
        #     grad_theta += 0
        grad_theta = (self.y - y_inp) @ X_inp
        grad_b = self.y - y_inp

        return grad_theta, grad_b

    def gradient_descent_logistic(
        self, alpha, num_pass, early_stop=0, standardized=True
    ):
        """Logistic Regression with gradient descent method

        :param alpha: (Hyper Parameter) Learning rate.
        :param num_pass: Number of iteration
        :param early_stop: (Hyper Parameter) Least improvement error allowed before stop.
            If improvement is less than the given value, then terminate the
            function and store the coefficents.  default = 0.
        :param standardized: bool, determine if we standardize the feature matrix.

        :returns self.theta: theta after training
        :returns self.b: b after training
        """

        if standardized:
            self.X = z_standardize(self.X)

        n, m = self.X.shape

        for i in tqdm(range(num_pass), desc="Training"):
            """
            TODO: 4. Modify the following code to implement gradient descent algorithm
            """
            grad_theta, grad_b = self.gradient(self.X, self.y, self.theta, self.b)
            temp_theta = self.theta - alpha * grad_theta
            temp_b = self.b - alpha * grad_b

            """
            TODO: 5. Modify the following code to implement early Stop
            Mechanism (use Logistic Loss when calculating error)
            """
            previous_y_hat = self.predict(self.X, self.theta)
            temp_y_hat = self.predict(self.X, temp_theta)
            pre_error = self.log_loss(previous_y_hat, self.y)
            temp_error = self.log_loss(temp_y_hat, self.y)
            print(pre_error, temp_error)
            if (abs(pre_error - temp_error) < early_stop) | (
                abs(abs(pre_error - temp_error) / pre_error) < early_stop
            ):
                return temp_theta, temp_b

            self.theta = temp_theta
            self.b = temp_b
        return self.theta, self.b

    def log_loss(self, pred: np.array, labels: np.array) -> float:
        """Calculate cross entropy loss.

        :param pred: predicted labels
        :param labels: actual labels
        :returns: cross entropy loss
        """
        return (labels * np.log(pred)) + ((1 - labels) * np.log(1 - pred))

    def predict_ind(self, x: np.array, weights: np.array) -> int:
        """Predict the most likely class label of one test instance.

        :param x: Matrix, array or list. Input feature point.
        :returns: prediction of given data point
        """
        return np.argmax(sigmoid(np.dot(x, weights)))

    def predict(self, X: np.ndarray, weights: np.array):
        """Predict for all data

        :param x: Matrix, array or list. Input feature point.
        :returns p: prediction of given data matrix
        """
        return np.array([self.predict_ind(vect, weights) for vect in X])

    def score(self, X: np.ndarray, y: np.array) -> float:
        """Calculate squared error.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: y for X
        :returns: square error of predictions
        """
        self.metric = "Square Error"
        square_error = sum(self.predict(X) - y)
        return square_error
