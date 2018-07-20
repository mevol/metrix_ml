#import packages
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

class SplitData(object):
    def __init__(self, df):
        self.df = df
    
    def split_data(df):
        X = df
        y = df['EP_success']
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_work = X_train.copy()
        return X_train_work, X_train, y_train, X_test, y_test

    #split_data(metrix)


