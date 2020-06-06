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
from modelzoo.models import InvalidModel, DecisionTree


MODELS = {
    "decision-tree": DecisionTree,
}
MODELS_HELP = ','.join(list(MODELS.keys()))


def load_data():
    url_Wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    wine = pd.read_csv(url_Wine, delimiter=';')
    X = np.array(wine)[:, :-1]
    y = np.array(wine)[:, -1]
    y = np.array(y.flatten())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


@click.command()
@click.option("-m", "--model-type", default=None, help=MODELS_HELP)
def main(model_type):
    X_train, X_test, y_train, y_test = load_data()

    # Load, initialize, and fit model
    model = MODELS.get(model_type, lambda: InvalidModel())
    clf = model()
    clf.fit(X_train, y_train)

    print("-"*10, clf, "-"*10)
    clf.score(X_train, y_train, datatype="Train")
    clf.score(X_test, y_test)


if __name__ == "__main__":
    main()
