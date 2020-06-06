class Model:
    def __repr__(self):
        return self.__class__.__name__


class InvalidModel:
    def __init__(self):
        raise SystemExit("No model or Invalid model provided")
