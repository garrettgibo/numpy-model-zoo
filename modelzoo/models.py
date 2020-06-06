"""
Custom Decision Tree
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from tqdm import tqdm


class Model(ABC):
    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def score(self, X, y):
        pass


class InvalidModel:
    def __init__(self):
        raise SystemExit("No model or Invalid model provided")


class DecisionTree(Model):
    """
    Decision Tree Classifier

    Attributes:
        root: Root Node of the tree.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split
        n_features: Number of features to use during building the tree.
            (Random Forest)
        n_split:  Number of split for each feature. (Random Forest)

    """

    def __init__(self, max_depth=1000, size_allowed=1, n_features=None, n_split=None):
        """Decision Tree initialization.

        :param max_depth: maximum amount of recursion for tree to create nodes
        :param size_allowed: minimum node size
        :param n_features: number of features to evaluate for decisions
        :param n_split: number of splits to do on features
        """
        self.root = 1
        self.max_depth = max_depth
        self.size_allowed = size_allowed
        self.n_features = n_features
        self.n_split = n_split

    class Node:
        """
        Node Class for the building the tree.

        Attribute:
            threshold: The threshold like if x1 < threshold, for
                spliting.
            feature: The index of feature on this current node.
            left: Pointer to the node on the left.
            right: Pointer to the node on the right.
            pure: Bool, describe if this node is pure.
            predict: Class, indicate what the most common Y on this
                node.
        """

        def __init__(self, threshold: float = None, feature: int = None):
            """Decision Tree Node initialization.

            :param threshold: threshold to split features on
            :param feature: feature index to split on
            """
            self.threshold = threshold
            self.feature = feature
            self.left = None
            self.right = None
            self.pure = False
            self.depth = 1
            self.predict = -1

    def entropy(self, y: np.array) -> float:
        """Calculate entropy for provided y.

        :param y: vector of y to calculate entropy on
        :returns: calculated entropy
        """
        entro = 0
        classes, counts = np.unique(y, return_counts=True)
        counts = counts / sum(counts)  # normalize counts to get prob of class
        for count in counts:
            if count == 0:
                continue
            entro -= count * np.log(count)
        return entro

    def information_gain(
        self, values: np.array, y: np.array, threshold: float
    ) -> float:
        """Calculate the information gain, by using entropy function.

        IG(Z) = H(X) - H(X|Z)

        :param values: single vector of values to calculate IG
        :param y: vector of all y
        :param threshold: threshold to calculate IG off of
        :returns: calculate IG based off information gain formula
        """
        left_side = values < threshold
        left_prop = len(values[left_side]) / len(values)
        right_prop = 1 - left_prop

        left_entropy = self.entropy(y[left_side])
        right_entropy = self.entropy(y[~left_side])

        return self.entropy(y) - (left_prop * left_entropy + right_prop * right_entropy)

    def find_rules(self, X: np.ndarray) -> np.ndarray:
        """Helper method to find the split rules.

        Splitting rules are found by finding all unique values in a feature,
        then finding all the midpoints for the unique values.

        :param X: matrix or 2-D numpy array, represnting training instances
        :returns: 2-D array of all possible splits for features
        """
        rules = []
        # transpose X to get features(columns)
        for feature in X.T:
            unique_values = np.unique(feature)
            mids = np.mean([unique_values[:-1], unique_values[1:]], axis=0)
            rules.append(mids)
        return rules

    def next_split(self, X: np.ndarray, y: np.array) -> Tuple[float, int]:
        """Helper method to find the split with most information.

        :param X: matrix or 2-D numpy array, represnting training instances
            Each training instance is a feature vector.
        :param y: label contains the corresponding y. There might be
            multiple (i.e., > 2) classes.
        """
        rules = self.find_rules(X)
        max_info = -1
        num_col = 1
        threshold = 1

        # when number of features wasn't set, use all features
        if self.n_features is None:
            index_col = np.arange(X.shape[1])
        else:
            if isinstance(self.n_features, int):
                num_index = self.n_features
            # if num of featuers is 'sqrt' use sqrt of total number of features
            elif isinstance(self.n_features, str):
                num_index = round(np.sqrt(X.shape[1]))
                np.random.seed()
                index_col = np.random.choice(X.shape[1], num_index, replace=False)

        # Moving through columns
        for i in index_col:
            count_temp_rules = len(rules[i])

            # when number of splits wasn't set, use all splits
            if self.n_split is None:
                index_rules = np.arange(count_temp_rules)
            else:
                if isinstance(self.n_split, int):
                    num_rules = self.n_split
                elif isinstance(self.n_split, str):
                    num_rules = round(np.sqrt(X.shape[0]))
                    if num_rules > count_temp_rules:
                        num_rules = count_temp_rules
                    np.random.seed()
                    index_rules = np.random.choice(
                        count_temp_rules, num_rules, replace=False
                    )

            # find split and threshold that results in maximum information gain
            for j in index_rules:
                info = self.information_gain(X.T[i], y, rules[i][j])
                if info > max_info:
                    max_info = info
                    num_col = i
                    threshold = rules[i][j]
        return threshold, num_col

    def build_tree(self, X: np.ndarray, y: np.array, depth: int) -> Node:
        """ Helper function for building the tree.

        :param X: full X set to train from
        :param y: full vector of y
        :returns: root Node
        """
        first_threshold, first_feature = self.next_split(X, y)
        current = self.Node(first_threshold, first_feature)

        # base case 1 to end build early
        if (
            depth > self.max_depth
            or first_feature is None
            or X.shape[0] == self.size_allowed
        ):
            current.predict = np.argmax(np.bincount(y))
            current.pure = True
            return current

        # base case 2: node has become a leaf
        if len(np.unique(y)) == 1:
            current.pure = True
            current.predict = y[0]
            return current

        # Find the left node index with feature i <= threshold
        # Right with feature i > threshold.
        left_index = X.T[first_feature] <= first_threshold
        right_index = X.T[first_feature] > first_threshold

        # base case 3: either side is empty
        if sum(left_index) == 0 or sum(right_index) == 0:
            # NOTE this is being set to the first label, but it may be better
            # to set this to the most common label
            current.predict = y[0]
            current.pure = True
            return current

        # recusively build rest of tree
        left_X, left_y = X[left_index, :], y[left_index]
        current.left = self.build_tree(left_X, left_y, depth + 1)

        right_X, right_y = X[right_index, :], y[right_index]
        current.right = self.build_tree(right_X, right_y, depth + 1)

        return current

    def fit(self, X: np.ndarray, y: np.array):
        """ Fits the Decision Tree model based on the training X.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: y for X. There might be multiple (i.e., > 2) classes.
        """
        y = y.astype(int)  # y need to have integer classes
        self.root = self.build_tree(X, y, 1)
        return self

    def ind_predict(self, vector: np.array) -> int:
        """Predict the most likely class label of one test instance.

        :param vector: single vector to predict
        :returns: class predicted
        """
        current = self.root
        while not current.pure:
            feature = current.feature
            threshold = current.threshold
            if vector[feature] < threshold:
                current = current.left
            else:
                current = current.right
        return current.predict

    def predict(self, X: np.ndarray) -> np.array:
        """Predict y for entire Xset.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :returns: predictions of all instances in a list.
        """
        return np.array([self.ind_predict(vect) for vect in X])

    def score(self, X: np.ndarray, y: np.array) -> float:
        """Wrapper around predict to also get accuracy.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: y for X. There might be multiple (i.e., > 2) classes.
        :returns: avg_accuracy of predictions
        """
        self.metric = "Accuracy"
        pred = self.predict(X)
        avg_accuracy = (pred == y).mean()
        return avg_accuracy


class RandomForest(Model):
    """RandomForest Classifier

    Attributes:
        n_trees: Number of trees.
        trees: List store each individule tree
        n_features: Number of features to use during building each individule tree.
        n_split: Number of split for each feature.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split
    """

    def __init__(
        self,
        n_trees=25,
        n_features="sqrt",
        n_split=None,
        max_depth=1000,
        size_allowed=1,
    ):
        """Random Forest initialization"""
        self.n_trees = n_trees
        self.trees = []
        self.n_features = n_features
        self.n_split = n_split
        self.max_depth = max_depth
        self.size_allowed = size_allowed

    def fit(self, X: np.ndarray, y: np.array) -> None:
        """ Fits the Random Forest model based on the training X.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: y for X. There might be multiple (i.e., > 2) classes.
        """
        y = y.astype(int)  # y need to have integer classes
        for i in tqdm(range(self.n_trees), desc="Fitting Forest"):
            np.random.seed()
            # initialize tree with all parameters from forest
            temp_clf = DecisionTree(
                max_depth=self.max_depth,
                size_allowed=self.size_allowed,
                n_features=self.n_features,
                n_split=self.n_split,
            )
            temp_clf.fit(X, y)
            self.trees.append(temp_clf)
        return self

    def ind_predict(self, vector: np.array) -> float:
        """Predict the most likely class label of one test instance.

        :param vector: single vector to predict
        :returns: class predicted
        """
        # predict using majority rule from doing predictions from all trees
        results = np.array([tree.ind_predict(vector) for tree in self.trees])
        y, counts = np.unique(results, return_counts=True)
        return y[np.argmax(counts)]

    def predict_all(self, X: np.ndarray) -> np.array:
        """Predict y for entire Xset.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :returns: predictions of all instances in a list.
        """
        return np.array([self.ind_predict(vect) for vect in X])

    def score(self, X: np.ndarray, y: np.array):
        """Wrapper around predict_all to also get accuracy.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: y for X. There might be multiple (i.e., > 2) classes.
        :returns: avg_accuracy of predictions
        """
        self.metric = "Accuracy"
        pred = self.predict_all(X)
        avg_accuracy = (pred == y).mean()
        return avg_accuracy


class LinearRegression(Model):
    def __init__(
        self,
        alpha: float = 1e-10,
        num_iter: int = 10000,
        early_stop: float = 1e-50,
        intercept: bool = True,
        init_weight: np.ndarray = None,
    ):
        """Linear Regression Initialization

        :param alpha: learning rate
        :param num_iter: number of iterations to update coefficient with training X
        :param early_stop: Constant control early_stop.
        :param intercept: Bool, If we are going to fit a intercept, default True.
        :param init_weight: Matrix (n x 1), input init_weight for testing.
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight  # For testing correctness.

    def fit(self, X: np.ndarray, y: np.array):
        """Save the datasets in our model, and perform gradient descent.

        :param X: Matrix or 2-D array. Input feature matrix.
        :param y: Matrix or 2-D array. Input target value.
        """
        self.X = X
        self.y = y.T

        if self.intercept:
            # add column of ones to left side of matrix
            self.X = np.hstack([np.ones((self.X.shape[0], 1)), self.X])

        self.coef = np.random.uniform(-1, 1, self.X.shape[1])
        self.gradient_descent()

    def square_error(self, y: np.array, y_hat: np.array) -> float:
        """Calculate loss as square error.

        error = ∑(Y_i − Y_i_hat)^2
        :param y: labels
        :param y_hat: predictions
        :returns: square error
        """
        return sum(np.square(y - y_hat))

    def gradient(self, y_hat: np.array) -> None:
        """Helper function to calculate the gradient respect to coefficient.

        :param y_hat:
        """
        self.grad_coef = (self.y - y_hat) @ self.X

    def gradient_descent(self):
        """Training function """
        self.loss = []

        for i in tqdm(range(self.num_iter), desc="Iterations"):
            preds_y_hat = np.array([self.coef.T @ vect for vect in self.X])
            pre_error = self.square_error(self.y, preds_y_hat)
            self.gradient(preds_y_hat)

            # delta rule to find new weights
            temp_coef = self.coef + self.alpha * self.grad_coef

            # predict and find loss from new coefficients
            preds_new = np.array([temp_coef.T @ vect for vect in self.X])
            current_error = self.square_error(self.y, preds_new)

            # This is the early stop, don't modify fllowing three lines.
            if (abs(pre_error - current_error) < self.early_stop) | (
                abs(abs(pre_error - current_error) / pre_error) < self.early_stop
            ):
                self.coef = temp_coef
                return self
            # adaptive learning rate
            if current_error <= pre_error:
                self.alpha *= 1.3
                self.coef = temp_coef
            else:
                self.alpha *= 0.9

            # track loss for future analysis
            self.loss.append(current_error)

            # print values a total of 1000 times during training process
            if i % (self.num_iter / 100) == 0:
                print("Iteration: " + str(i))
                print("Coef: " + str(self.coef))
                print("Loss: " + str(current_error))
        return self

    def ind_predict(self, x: np.array) -> float:
        """Predict the value based on its feature vector x.

        :param x: Matrix, array or list. Input feature point.
        :returns: prediction of given X point
        """
        return self.coef.T @ x

    def predict(self, X: np.ndarray) -> np.array:
        """Predict value for all X

        :param X: matrix/2-D numpy array, represnting testing instances.
        :returns: prediction of given X matrix
        """
        if self.intercept:
            # add column of ones to left side of matrix
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.array([self.ind_predict(vect) for vect in X])

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
