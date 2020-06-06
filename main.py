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
from modelzoo.decision_tree import DecisionTree


MODELS = {
    "decision-tree": DecisionTree,
}


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


def evaluate_model(model, X_train, X_test, y_train, y_test):
    print("-"*10, model, "-"*10)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    avg_train = (pred_train == y_train).mean()
    avg_test = (pred_test == y_test).mean()

    print(f"Train Error: {avg_train}")
    print(f"Test Error: {avg_test}")


@click.command()
@click.argument("model_type", default=None)
def main(model_type):
    X_train, X_test, y_train, y_test = load_data()

    # Load, initialize, and fit model
    model = MODELS.get(model_type, lambda: "Invalid Model")
    clf = model()
    clf.fit(X_train, y_train)

    evaluate_model(clf, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
