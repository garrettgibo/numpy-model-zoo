"""
This model zoo is my implementation of the following models:
    - decision tree
    - random forest
    - linear regression
    - logistic regression
    - matrix factorization
    - naive bayes
    - kmeans
I have provided a main driver to test these models and compare them against
similar models from sklearn
"""
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modelzoo.models import (
    InvalidModel,
    DecisionTree,
    RandomForest,
    LinearRegression,
)


MODELS = {
    "decision-tree": DecisionTree,
    "random-forest": RandomForest,
    "linear-regression": LinearRegression
}
MODELS_HELP = ','.join(list(MODELS.keys()))


def load_data(test_size):
    url_Wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    wine = pd.read_csv(url_Wine, delimiter=';')
    X = np.array(wine)[:, :-1]
    y = np.array(wine)[:, -1]
    y = np.array(y.flatten()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    return X_train, X_test, y_train, y_test


@click.command()
@click.option("-m", "--model-type", default=None, help=MODELS_HELP)
def main(model_type):
    # load data and do a train test split
    X_train, X_test, y_train, y_test = load_data(test_size=0.2)

    model = MODELS.get(model_type, lambda: InvalidModel())
    clf = model()  # initialize model with default parameters
    clf.fit(X_train, y_train)

    print("-" * 10, clf, "-" * 10)
    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    print(f"{clf} - Train {clf.metric} -> {score_train}")
    print(f"{clf} - Test {clf.metric} -> {score_test}")


if __name__ == "__main__":
    main()
