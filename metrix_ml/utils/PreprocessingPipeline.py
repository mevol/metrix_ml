#import packages
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import DataFrameSelector


'''This is very basic code; need to turn this into a class
with functions; failed so far'''

attr = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                 'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                 'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                 'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF']

 
pipeline = Pipeline([
            ('selector', DataFrameSelector(attr)),
            ('imputer', Imputer(strategy='median')),
#            ('pandas_transform', PandasTransform(ColumnTransformation()))
#            ('transformer', ColumnTransformation()),
            ('std_scaler', StandardScaler())
        ])

pipeline.fit_transform(metrix)

