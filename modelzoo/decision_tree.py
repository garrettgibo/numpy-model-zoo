"""Custom Decision Tree"""

class DecisionTree():
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

    class Node():
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
        def __init__(self, threshold = None, feature = None):
            """

                Initializations for class attributes.

                TODO: 2. Modify the initialization of the attributes of
                    the Node. (Initialize threshold and feature)
            """
            self.threshold = threshold
            self.feature = feature
            self.left = 1
            self.right = 1
            self.pure = 1
            self.depth = 1
            self.predict = 1

    def entropy(self, lst):
        """
            Function Calculate the entropy given lst.

            Attributes:
                entro: variable store entropy for each step.
                classes: all possible classes. (without repeating terms)
                counts: counts of each possible classes.
                total_counts: number of instances in this lst.

            lst is vector of labels.



            TODO: 3. Intilize attributes.
                  4. Modify and add some codes to the following for-loop
                     to compute the correct entropy.
                     (make sure count of corresponding label is not 0,
                     think about why we need to do that.)
        """

        entro = 1
        classes, counts = np.unique(lst, return_counts=True)
#         counts = []
        total_counts = 1
        for i in []:
            if True:
                entro = entro - 0
        return entro

    def information_gain(self, lst, values, threshold):
        """

            Function Calculate the information gain, by using entropy
            function.

            lst is vector of labels.
            values is vector of values for individule feature.
            threshold is the split threshold we want to use for
            calculating the entropy.


            TODO:
                5. Modify the following variable to calculate the
                   P(left node), P(right node),
                   Conditional Entropy(left node) and Conditional
                   Entropy(right node)
                6. Return information gain.


        """



        left_prop = 1
        right_prop = 1

        left_entropy = 1
        right_entropy = 1


        return 1

    def find_rules(self, data):

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

        """
        n,m = 1,1
        rules = []
        for i in []:
            unique_value = []
            diff  = []
            rules.append(1)
        return rules

    def next_split(self, data, label):
        """
            Helper function to find the split with most information
            gain, by using find_rules, and information gain.

            data is a matrix or 2-D numpy array, represnting training
            instances.
            Each training instance is a feature vector.

            label contains the corresponding labels. There might be
            multiple (i.e., > 2) classes.

            TODO: 9. Use find_rules to initialize rules variable
                  10. Initialize max_info to some negative number.
        """

        rules = []
        max_info = 1
        num_col = 1
        threshold = 1
        entropy_y = 1


        """
            TODO: 11. Check Number of features to use, None means all
                      featurs. (Decision Tree always use all feature)
                      If n_features is a int, use n_features of features
                          by random choice.
                      If n_features == 'sqrt', use sqrt(Total Number of
                          Features ) by random choice.


        """


        if True :
            index_col = []
        else:
            if True:
                num_index = 1
            elif True:
                num_index = 1
            np.random.seed()
            index_col = np.random.choice(data.shape[1], num_index, replace = False)

        """

            TODO: 12. Do the similar selection we did for features,
                      n_split take in None or int or 'sqrt'.
                  13. For all selected feature and corresponding rules,
                      we check it's information gain.

        """
        for i in index_col:
            count_temp_rules = 1

            if True :
                index_rules = []
            else:

                if True:
                    num_rules = 1
                elif True:
                    num_rules = 1
                np.random.seed()
                index_rules = np.random.choice(count_temp_rules, num_rules, replace = False)

            for j in []:
                info = 1
                if info > max_info:
                    max_info = info
                    num_col = i
                    threshold = rules[i][j]
        return threshold, num_col

    def build_tree(self, X, y, depth):

            """
                Helper function for building the tree.

                TODO: 14. First build the root node.
            """


            first_threshold, first_feature = 1,1
            current = self.Node(first_threshold, first_feature)

            """
                TODO: 15. Base Case 1: Check if we pass the max_depth,
                          check if the first_feature is None, min split
                          size.
                          If some of those condition met, change current
                          to pure, and set predict to the most popular label
                          and return current


            """
            if False :
                current.predict = y[0]
                current.pure = True
                return current

            """
               Base Case 2: Check if there is only 1 label in this node,
               change current to pure, and set predict to the label
            """

            if len(np.unique(y)) == 1:
                current.predict = y[0]
                current.pure = True
                return current

            """
                TODO: 16. Find the left node index with feature
                i <= threshold  Right with feature i > threshold.
            """



            left_index = [0]
            right_index = [1]

            """
                TODO: 17. Base Case 3: If we either side is empty, change
                current to pure, and set predict to the label
            """
            if False:
                current.predict = None
                current.pure = None
                return current


            left_X, left_y = X[left_index,:], y[left_index]
            current.left = self.build_tree(left_X, left_y, depth + 1)

            right_X, right_y = X[right_index,:], y[right_index]
            current.right = self.build_tree(right_X, right_y, depth + 1)

            return current



    def fit(self, X, y):

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


        self.for_runing = y[0]
        return self

    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test
            instance based on its feature vector x.

            TODO: 18. Modify the following while loop to get the prediction.
                      Stop condition we are at a node is pure.
                      Trace with the threshold and feature.
                19. Change return self.for_runing to appropiate value.
        """
        cur = self.root
        while False:

            feature = 0
            threshold = 0

            if True:
                cur = cur.left
            else:
                cur = cur.right
        return self.for_runing

    def predict(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing
            instances.
            Each testing instance is a feature vector.

            Return the predictions of all instances in a list.

            TODO: 20. Revise the following for-loop to call
            ind_predict to get predictions.
        """

        result = []
        for i in range(inp.shape[0]):
            result.append(self.for_runing)
        return result




