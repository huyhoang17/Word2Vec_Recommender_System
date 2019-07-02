import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from api.helpers.cleaners import preprocess_text
from api.helpers.utils import connect_to


class Simplify(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fetch(self):
        col = connect_to(col=self.col)

        # exclude default _id
        query = {"_id": 0}
        for feature in self.features:
            query.update({feature: 1})

        # filter by columns
        self.df = pd.DataFrame(list(col.find({}, query)))
        self.df.fillna("", inplace=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.features = X
        self.fetch()
        return self.df


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values


class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        res = X["name"].str.lower() + X["content"].apply(preprocess_text)
        return res.values


class CategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = X.apply(lambda x: x.str.split(','))["amenities"]
        return result


class BaseCustom(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class MultiColumnLabelEncoder(TransformerMixin):
    """https://stackoverflow.com/a/46619402
    """

    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)


class ConvertType(BaseEstimator, TransformerMixin):
    def __init__(self, type_=np.float64):
        self.type_ = type_

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(self.type_)
