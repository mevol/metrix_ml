#import packages
import pandas as pd
import os
import numpy as np
import ColumnTransformation
import DataFrameSelector
import PandasFeatureUnion
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler



class PreprocessingPipeline():
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform():
        attr = list(metrix)
        pipeline = PandasFeatureUnion([
                        Pipeline([
                            ('selector', DataFrameSelector(attr))
                        ])
        ])


        
#    full_pipeline = FeatureUnion(standard_pipeline)
transform.fit_transform(metrix)