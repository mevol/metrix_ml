import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class LoadPrepareInput(object):
    '''A class to define input and output directories and load CSV file
    containing the data as a table as well as writing CSV files after
    transformations'''
    def __init__(self):
        pass
    
    #I should think about providing input/outputdirectories and paths
    #as input to the function
    datestring = datetime.strftime(datetime.now(), '%Y%m%d')
    METRIX_PATH = "/Users/melanievollmar/Documents/METRICS/database_output_analysis/metrix_db_20170531/Python_ML/"
    INPUT_PATH = os.path.join(METRIX_PATH, "data")
    OUTPUT_PATH = METRIX_PATH        
    if not os.path.exists(os.path.join(METRIX_PATH, datestring)):
        os.makedirs(os.path.join(OUTPUT_PATH, datestring))
    OUTPUT_DIR = os.path.join(OUTPUT_PATH, datestring) 
    
    def load_data(filename, input_path = INPUT_PATH):
        '''load CSV file to be a dataframe'''
        csv_path = os.path.join(input_path, filename)
        return pd.read_csv(csv_path)

    def write_data(df, output_path = OUTPUT_PATH):
        '''write dataframe to CSV file'''
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        csv_path = os.path.join(output_path, 'df'+datestring+'.csv')
        return df.to_csv(csv_path)

    
#####################################################################################    
    #make_directories()
    #dummy = load_data('May_2017_combined_valid_results_EP-Shelx_fail_removed.csv')
    #print(dummy)
    #write_data(dummy)