#import packages
import pandas as pd
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
		'''A class to help select a dataframe columns to use in
		an input preparation pipeline if not the entire set is
		wanted; selection based on a list of column labels which
		gets passed into the class'''
    def __init__(self, attr):
        self.attr = attr
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attr].values

#DataFrameSelector(list(metrix))

