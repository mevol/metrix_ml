import pickle
import argparse
#import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='Prediction using trained model')
  
  parser.add_argument(
    '--input', 
    type=str, 
    dest="input",
    default="",
    help='The input CSV file')

  parser.add_argument(
    '--model', 
    type=str, 
    dest="model",
    default="",
    help='The trained classifier model')   
    
  parser.add_argument(
    '--outdir',
    type=str,
    dest='outdir',
    default='',
    help='Specify output directory')

  args = parser.parse_args()
  if args.input == '':
    parser.print_help()
    exit(0)
  return args

###############################################################################
#
#  load the data from CSV file and creating output directory
#
###############################################################################

def load_unknown_data(csv_path):
  '''load the raw data as stored in CSV file'''
#  return pd.read_csv(csv_path)
  return read_csv(csv_path)

def load_pickle(filename):
  with open(filename, 'rb') as f:
    model = joblib.load(f)
  return model

def make_output_folder(outdir):
  output_dir = os.path.join(outdir, 'predictions')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class for ML using random forest with randomised search
#
###############################################################################

class PredictUnknown(object):
  '''This class is the doing the actual work in the following steps:
     * column transformation and standardising data
     * prediction using loaded model
  '''
  def __init__(self, data, model, output_dir):
    self.data=data
    self.output_dir=output_dir
    self.model=model   
    self.prepare_data()
    self.predict()

  ###############################################################################
  #
  #  create input data frame for selected features according to Feature5
  #
  ###############################################################################

  def prepare_data(self):
    '''Function carrying out custom column transformations and to standardis
    samples for SVM.
    '''
    print('*' *80)
    print('*    Preparing input data to match Feature5')
    print('*' *80)

    #database plus manually added data
    data_initial = ['eLLG', 'seq_ident', 'MW_chain']

    data_initial = self.data[data_initial]  
    X_data_initial = data_initial.fillna(0)
    self.X_data_initial = X_data_initial
    
#    print(self.X_data_initial)
   
  ###############################################################################
  #
  #  predict using Feature5 data frame
  #
  ###############################################################################

  def predict(self):
    '''Function to predict the likely experimental phasing outcome using a
    trained and saved model
    '''
    print('*' *80)
    print('*    Using trained model to predict results')
    print('*' *80)

    unknown = self.X_data_initial.values

    for line in unknown:
      y_pred = self.model.predict(line.reshape(1, -1))
      y_pred_proba = self.model.predict_proba(line.reshape(1, -1))
      fail_prob = round(y_pred_proba[0][0], 4) * 100
      succ_prob = round(y_pred_proba[0][1], 4) * 100
      y_pred_adj = [1 if x >= 0.9317 else 0 for x in y_pred_proba[:, 1]]

      with open(os.path.join(self.output_dir, 'results_predict.txt'), 'a') as text_file:
        #text_file.write('Experimental phasing outcome: %s \n' %y_pred)
        text_file.write('Probability for experimental phasing outcome: \n')
        text_file.write('Failure: %.2f \n' %fail_prob)
        text_file.write('Success: %.2f \n' %succ_prob)
        #text_file.write('Predicted class after applying threshold 93.17%% for class 1: %s \n' %str(y_pred_adj))
        text_file.write('*' * 80)
        text_file.write('\n')
      
      #print('Experimental phasing outcome: %s' %y_pred)
      print('Probability for experimental phasing outcome:')
      print('Failure: %s' %fail_prob)
      print('Success: %s' %succ_prob)
      #print('Predicted class after applying threshold 93.17%% for class 1: %s \n' %str(y_pred_adj))
      print('*' * 80)

  

    #make predictions
    ynew = self.model.predict(unknown)
    # show the inputs and predicted outputs
    for i in range(len(unknown)):
      print("X=%s, Predicted=%s" % (unknown[i], ynew[i]))

      with open(os.path.join(self.output_dir, 'results_predict.txt'), 'a') as text_file:
        text_file.write("X=%s, Predicted=%s \n" % (unknown[i], ynew[i]))

  
    # make probabilistic predictions
    ynew_proba = self.model.predict_proba(unknown)
    # show the inputs and predicted probabilities
    for i in range(len(unknown)):
      print("X=%s, Predicted=%s" % (unknown[i], ynew_proba[i]))

      with open(os.path.join(self.output_dir, 'results_predict.txt'), 'a') as text_file:
        text_file.write("X=%s, Predicted=%s \n" % (unknown[i], ynew_proba[i]))


  
def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  data = load_unknown_data(args.input)
  model = load_pickle(args.model)

  predict = make_output_folder(args.outdir)

  ###############################################################################

  predict_unknown = PredictUnknown(data, model, predict)

