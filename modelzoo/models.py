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
    def score(self, data, labels):
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

    def entropy(self, labels: np.array) -> float:
        """Calculate entropy for provided labels.

        :param labels: vector of labels to calculate entropy on
        :returns: calculated entropy
        """
        entro = 0
        classes, counts = np.unique(labels, return_counts=True)
        counts = counts / sum(counts)  # normalize counts to get prob of class
        for count in counts:
            if count == 0:
                continue
            entro -= count * np.log(count)
        return entro

    def information_gain(
        self, values: np.array, labels: np.array, threshold: float
    ) -> float:
        """Calculate the information gain, by using entropy function.

        IG(Z) = H(X) - H(X|Z)

        :param values: single vector of values to calculate IG
        :param labels: vector of all labels
        :param threshold: threshold to calculate IG off of
        :returns: calculate IG based off information gain formula
        """
        left_side = values < threshold
        left_prop = len(values[left_side]) / len(values)
        right_prop = 1 - left_prop

        left_entropy = self.entropy(labels[left_side])
        right_entropy = self.entropy(labels[~left_side])

        return self.entropy(labels) - (
            left_prop * left_entropy + right_prop * right_entropy
        )

    def find_rules(self, data: np.ndarray) -> np.ndarray:
        """Helper method to find the split rules.

        Splitting rules are found by finding all unique values in a feature,
        then finding all the midpoints for the unique values.

        :param data: matrix or 2-D numpy array, represnting training instances
        :returns: 2-D array of all possible splits for features
        """
        rules = []
        # transpose data to get features(columns)
        for feature in data.T:
            unique_values = np.unique(feature)
            mids = np.mean([unique_values[:-1], unique_values[1:]], axis=0)
            rules.append(mids)
        return rules

    def next_split(self, data: np.ndarray, labels: np.array) -> Tuple[float, int]:
        """Helper method to find the split with most information.

        :param data: matrix or 2-D numpy array, represnting training instances
            Each training instance is a feature vector.
        :param labels: label contains the corresponding labels. There might be
            multiple (i.e., > 2) classes.
        """
        rules = self.find_rules(data)
        max_info = -1
        num_col = 1
        threshold = 1

        # when number of features wasn't set, use all features
        if self.n_features is None:
            index_col = np.arange(data.shape[1])
        else:
            if isinstance(self.n_features, int):
                num_index = self.n_features
            # if num of featuers is 'sqrt' use sqrt of total number of features
            elif isinstance(self.n_features, str):
                num_index = round(np.sqrt(data.shape[1]))
                np.random.seed()
                index_col = np.random.choice(data.shape[1], num_index, replace=False)

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
                    num_rules = round(np.sqrt(data.shape[0]))
                    if num_rules > count_temp_rules:
                        num_rules = count_temp_rules
                    np.random.seed()
                    index_rules = np.random.choice(
                        count_temp_rules, num_rules, replace=False
                    )

            # find split and threshold that results in maximum information gain
            for j in index_rules:
                info = self.information_gain(data.T[i], labels, rules[i][j])
                if info > max_info:
                    max_info = info
                    num_col = i
                    threshold = rules[i][j]
        return threshold, num_col

    def build_tree(self, X: np.ndarray, y: np.array, depth: int) -> Node:
        """ Helper function for building the tree.

        :param X: full data set to train from
        :param y: full vector of labels
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
        """ Fits the Decision Tree model based on the training data.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: labels for data. There might be multiple (i.e., > 2) classes.
        """
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
        """Predict labels for entire dataset.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: labels for data. There might be multiple (i.e., > 2) classes.
        :returns: predictions of all instances in a list.
        """
        return np.array([self.ind_predict(vect) for vect in X])

    def score(self, data: np.ndarray, labels: np.array, datatype="Test") -> float:
        """Wrapper around predict to also get accuracy.

        :param data: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param labels: labels for data. There might be multiple (i.e., > 2) classes.
        :returns: avg_accuracy of predictions
        """
        pred = self.predict(data)
        avg_accuracy = (pred == labels).mean()
        print(f"{datatype} accuracy: {avg_accuracy}")
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
        """ Fits the Random Forest model based on the training data.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: labels for data. There might be multiple (i.e., > 2) classes.
        """
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
        labels, counts = np.unique(results, return_counts=True)
        return labels[np.argmax(counts)]

    def predict_all(self, X: np.ndarray) -> np.array:
        """Predict labels for entire dataset.

        :param X: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param y: labels for data. There might be multiple (i.e., > 2) classes.
        :returns: predictions of all instances in a list.
        """
        return np.array([self.ind_predict(vect) for vect in X])

    def score(self, data: np.ndarray, labels: np.array, datatype="Test"):
        """Wrapper around predict_all to also get accuracy.

        :param data: matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.
        :param labels: labels for data. There might be multiple (i.e., > 2) classes.
        :returns: avg_accuracy of predictions
        """
        pred = self.predict_all(data)
        avg_accuracy = (pred == labels).mean()
        print(f"{datatype} accuracy: {avg_accuracy}")
        return avg_accuracy
