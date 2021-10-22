import pickle
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='Prediction using trained model')
  
  parser.add_argument(
    '--data', 
    type=str, 
    dest="data",
    default="",
    help='The calibration data CSV file')

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
  if args.data == '':
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
  return pd.read_csv(csv_path)

def load_pickle(filename):
  with open(filename, 'rb') as f:
    model = joblib.load(f)
  return model

def make_output_folder(outdir):
  names = ['calibrate', 'bbbb']
  result = []
  for name in names:
    name = os.path.join(outdir, 'calibrate', name)
    os.makedirs(name, exist_ok=True)
    result.append(name)
  return result

###############################################################################
#
#  class for ML using random forest with randomised search
#
###############################################################################

class Calibrate(object):
  '''This class is the doing the actual work in the following steps:
     * column transformation and standardising data
     * prediction using loaded model
  '''
  def __init__(self, data, model, calibrate, bbbb):
    self.data=data
    self.calibrate=calibrate
    self.model=model   
    self.prepare_data()
    self.calibration()

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
    print('*    Spliting calibration data into y and X')
    print('*' *80)

    #database plus manually added data
    data_initial = ['lowreslimit', 'anomalousslope', 'anomalousCC', 'diffI', 'diffF', 'f']

    y = self.data['EP_success']
    self.y = y
    print(self.y)
    
    with open(os.path.join(self.calibrate, 'calibrate.txt'), 'a') as text_file:
      text_file.write(str(self.y))
      text_file.write('\n')

    data_initial = self.data[data_initial]  
    X_data_initial = data_initial.fillna(0)
    self.X_data_initial = X_data_initial
    
    print(self.X_data_initial)
    
    with open(os.path.join(self.calibrate, 'calibrate.txt'), 'a') as text_file:
      text_file.write(str(self.X_data_initial))
      text_file.write('\n')
      text_file.write('Split calibration data into y and X \n')

   
  ###############################################################################
  #
  #  predict using Feature5 data frame
  #
  ###############################################################################

  def calibration(self):
    '''Function to predict the likely experimental phasing outcome using a
    trained and saved model
    '''
    print('*' *80)
    print('*    Calibrating pre-trained model')
    print('*' *80)

    clf_cccv = CalibratedClassifierCV(self.model, cv='prefit')
    self.calibrated_clf_cccv = clf_cccv.fit(self.X_data_initial, self.y)

    def write_pickle(forest, directory):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      joblib.dump(forest, os.path.join(directory,'calibrated_classifier_'+datestring+'.pkl'))
      with open(os.path.join(directory, 'calibrate.txt'), 'a') as text_file:
        text_file.write('Creating pickle file for for calibrated classifier \n')
    
    write_pickle(self.calibrated_clf_cccv, self.calibrate)

    cal_acc = clf_cccv.score(self.X_data_initial, self.y)

    print(cal_acc)

    with open(os.path.join(self.calibrate, 'calibrate.txt'), 'a') as text_file:
      text_file.write(str(cal_acc))
      text_file.write('\n')
  
def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  data = load_unknown_data(args.data)
  model = load_pickle(args.model)

  calibrate, bbbb= make_output_folder(args.outdir)

  ###############################################################################

  calibrate = Calibrate(data, model, calibrate, bbbb)

