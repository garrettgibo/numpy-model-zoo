# Custom Model Zoo ![Style/Formatting CI](https://github.com/garrettgibo/custom-model-zoo/workflows/Style/Formatting%20CI/badge.svg)

## Prerequisites

It is recommended to use [pipenv](https://pipenv.pypa.io/en/latest/) to manage
the virtual environment for this project.

With this command your python environment should be setup with everything you
need to run/test all models.

```Shell
pipenv install
```

## Usage

The model comparison notebook can be used to verify custom model performance
against sklearn. Running the notebook in the pipenv virtual environment will
ensure that all depenedencies are installed when testing.

```Shell
pipenv run notebook
```
