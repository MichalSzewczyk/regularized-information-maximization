import pandas as pd


class ClassificationDTO:
    def __init__(self, X, alpha):
        self.X = X
        self.alpha = alpha


def load_data(data_source):
    user_knowledge = pd.read_csv(data_source).values
    return user_knowledge


loaded_data = load_data('sample_data.csv')
examples = loaded_data[:, :-1].astype(float)
real_classification = loaded_data[:, -1]


