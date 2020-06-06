"""
Custom Decision Tree
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


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
        """
        Initializations for class attributes.

        TODO: 1. Modify the initialization of the attributes of the
        Decision Tree classifier
        DONE
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

        def __init__(self, threshold=None, feature=None):
            """
            Initializations for class attributes.

            TODO: 2. Modify the initialization of the attributes of
                the Node. (Initialize threshold and feature)
            DONE
            """
            self.threshold = threshold
            self.feature = feature
            self.left = None
            self.right = None
            self.pure = False
            self.depth = 1
            self.predict = -1

    def entropy(self, labels: np.array) -> float:
        """
        Function Calculate the entropy given labels.

        Attributes:
            entro: variable store entropy for each step.
            classes: all possible classes. (without repeating terms)
            counts: counts of each possible classes.
            total_counts: number of instances in this labels.

        labels is vector of labels.



        TODO: 3. Intilize attributes.
            DONE
              4. Modify and add some codes to the following for-loop
                 to compute the correct entropy.
                 (make sure count of corresponding label is not 0,
                 think about why we need to do that.)
            DONE
        """

        entro = 0
        classes, counts = np.unique(labels, return_counts=True)
        # total_counts = sum(counts)
        counts = counts / sum(counts)  # normalize counts
        for count in counts:
            if count == 0:
                continue
            entro -= count * np.log(count)
        return entro

    def information_gain(
        self, values: np.array, labels: np.array, threshold: float
    ) -> float:
        """
        Function Calculate the information gain, by using entropy
        function.

        labels is vector of labels.D
        values is vector of values for individule feature.
        threshold is the split threshold we want to use for
        calculating the entropy.


        TODO:
            5. Modify the following variable to calculate the
               P(left node), P(right node),
               Conditional Entropy(left node) and Conditional
               Entropy(right node)
            DONE
            6. Return information gain.
            DONE

        IG(Z) = H(X) - H(X|Z)
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
        """
        Helper function to find the split rules.

        data is a matrix or 2-D numpy array, represnting training
        instances.
        Each training instance is a feature vector.

        TODO: 7. Modify the following for loop, which loop through
                 each column(feature).
                 Find the unique value of each feature, and find
                 the mid point of each adjacent value.
              8. Store them in a list, return that list.
              DONE
        """
        # n, m = 1, 1
        rules = []
        # transpose data to get features(columns)
        for feature in data.T:
            unique_values = np.unique(feature)
            # diff = []
            midpoints = np.mean([unique_values[:-1], unique_values[1:]], axis=0)
            rules.append(midpoints)
        return rules

    def next_split(self, data: np.ndarray, labels: np.array) -> Tuple[float, int]:
        """
        Helper function to find the split with most information
        gain, by using find_rules, and information gain.

        data is a matrix or 2-D numpy array, represnting training
        instances.
        Each training instance is a feature vector.

        label contains the corresponding labels. There might be
        multiple (i.e., > 2) classes.

        TODO: 9. Use find_rules to initialize rules variable
        DONE
              10. Initialize max_info to some negative number.
              DONE
        """
        rules = self.find_rules(data)
        max_info = -1
        num_col = 1
        threshold = 1

        """
        TODO: 11. Check Number of features to use, None means all
                  featurs. (Decision Tree always use all feature)
                  DONE
                  If n_features is a int, use n_features of features
                      by random choice.
                  If n_features == 'sqrt', use sqrt(Total Number of
                      Features ) by random choice.
                  DONE
        """
        if self.n_features is None:
            index_col = np.arange(data.shape[1])
        else:
            if isinstance(self.n_features, int):
                num_index = self.n_features
            elif isinstance(self.n_features, str):
                num_index = np.sqrt(self.n_features)
                np.random.seed()
                index_col = np.random.choice(data.shape[1], num_index, replace=False)

        """
        TODO: 12. Do the similar selection we did for features,
                  n_split take in None or int or 'sqrt'.
                  DONE
              13. For all selected feature and corresponding rules,
                  we check it's information gain.
        """
        # Moving through columns
        for i in index_col:
            count_temp_rules = len(rules[i])

            # determine index splits for each row
            if self.n_split is None:
                index_rules = np.arange(data.shape[1])
                index_rules = np.arange(count_temp_rules)
            else:
                if isinstance(self.n_split, int):
                    num_rules = self.n_split
                elif isinstance(self.n_split, str):
                    num_rules = np.sqrt(self.n_split)
                    np.random.seed()
                    index_rules = np.random.choice(
                        count_temp_rules, num_rules, replace=False
                    )

            for j in index_rules:
                info = self.information_gain(data.T[i], labels, rules[i][j])
                if info > max_info:
                    max_info = info
                    num_col = i
                    threshold = rules[i][j]
        return threshold, num_col

    def build_tree(self, X: np.ndarray, y: np.array, depth: int) -> Node:
        """
            Helper function for building the tree.

            TODO: 14. First build the root node.
        """
        first_threshold, first_feature = self.next_split(X, y)
        current = self.Node(first_threshold, first_feature)

        """
            TODO: 15. Base Case 1: Check if we pass the max_depth,
                      check if the first_feature is None, min split
                      size.
                      If some of those condition met, change current
                      to pure, and set predict to the most popular label
                      and return current
                    DONE?


        """
        if depth > self.max_depth or first_feature is None:
            current.predict = np.argmax(np.bincount(y))
            current.pure = True
            return current

        """
           Base Case 2: Check if there is only 1 label in this node,
           change current to pure, and set predict to the label
        """
        if len(np.unique(y)) == 1:
            current.pure = True
            current.predict = y[0]
            return current

        """
            TODO: 16. Find the left node index with feature
            i <= threshold  Right with feature i > threshold.
        """
        left_index = X.T[first_feature] <= first_threshold
        right_index = X.T[first_feature] > first_threshold

        """
            TODO: 17. Base Case 3: If we either side is empty, change
            current to pure, and set predict to the label
            DONE
        """
        if sum(left_index) == 0 or sum(right_index) == 0:
            # NOTE this is being set to the first label, but it may be better to
            # set this to the most common label
            current.predict = y[0]
            current.pure = True
            return current

        left_X, left_y = X[left_index, :], y[left_index]
        current.left = self.build_tree(left_X, left_y, depth + 1)

        right_X, right_y = X[right_index, :], y[right_index]
        current.right = self.build_tree(right_X, right_y, depth + 1)

        return current

    def fit(self, X: np.ndarray, y: np.array) -> None:
        """
            The fit function fits the Decision Tree model based on
            the training data.

            X_train is a matrix or 2-D numpy array, represnting
            training instances.
            Each training instance is a feature vector.

            y_train contains the corresponding labels. There might
            be multiple (i.e., > 2) classes.
        """
        self.root = self.build_tree(X, y, 1)
        # self.for_runing = y[0]
        # return self

    def ind_predict(self, vector: np.array) -> int:
        """
            Predict the most likely class label of one test
            instance based on its feature vector x.

            TODO: 18. Modify the following while loop to get the prediction.
                      Stop condition we are at a node is pure.
                      Trace with the threshold and feature.
                      DONE
                19. Change return self.for_runing to appropiate value.
                DONE
        """
        current = self.root
        while not current.pure:
            feature = current.feature
            threshold = current.threshold
            current = current.left if vector[feature] < threshold else current.right
        return current.predict

    def predict(self, X: np.ndarray) -> np.array:
        """
            X is a matrix or 2-D numpy array, represnting testing
            instances.
            Each testing instance is a feature vector.

            Return the predictions of all instances in a list.

            TODO: 20. Revise the following for-loop to call
            ind_predict to get predictions.
            DONE
        """
        result = np.array([self.ind_predict(vect) for vect in X])
        return result

    def score(self, data: np.ndarray, labels: np.array, datatype="Test"):
        pred = self.predict(data)
        avg_accuracy = (pred == labels).mean()
        print(f"{datatype} accuracy: {avg_accuracy}")
