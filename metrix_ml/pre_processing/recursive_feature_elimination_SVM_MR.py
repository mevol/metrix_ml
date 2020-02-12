###############################################################################
#
#  imports and set up environment
#
###############################################################################
'''Defining the environment for this class'''
import argparse
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from datetime import datetime

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='recursive feature elimination')
  
  parser.add_argument(
    '--input', 
    type=str, 
    dest="input",
    default="",
    help='The input CSV file')
    
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
#  load the data from CSV file
#
###############################################################################

def load_metrix_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path)

def make_output_folder(outdir):
  output_dir = os.path.join(outdir, 'recursive_feature_elimination')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class for recursive feature elimination
#
###############################################################################

class RecursiveFeatureElimination(object):
  '''This class is the doing the actual work in the following steps:
     * define smaller data frames: database, man_add, transform
     * split the data into training and test set
     * setup and run a grid search for best paramaters to define a random forest
     * create a new random forest with best parameters
     * predict on this new random forest with test data and
       cross-validated training data
     * analyse the predisctions with graphs and stats
  '''
  def __init__(self, metrix, output_dir):
    self.metrix = metrix
    self.output_dir = output_dir
    self.prepare_metrix_data()
    self.split_data()
    self.run_rfecv()

###############################################################################
#
#  creating smaller data frame
#
###############################################################################

  def prepare_metrix_data(self):
    '''Function to create smaller dataframe with columns of interest.
    ******
    Input: large data frame
    Output: smaller dataframe
    '''
    print('*' *80)
    print('*    Preparing input dataframe X_metrix')
    print('*' *80)

    #database plus manually added data
    self.X_metrix = self.metrix[['IoverSigma', 'completeness', 'RmergeI',
                    'lowreslimit', 'RpimI', 'multiplicity', 'RmeasdiffI',
                    'wilsonbfactor', 'RmeasI', 'highreslimit', 'RpimdiffI', 
                    'RmergediffI', 'totalobservations', 'cchalf', 'totalunique',
                    'mr_reso', 'eLLG', 'tncs', 'seq_ident', 'model_res',
                    'No_atom_chain', 'MW_chain', 'No_res_chain', 'No_res_asu',
                    'likely_sg_no', 'xia2_cell_volume', 'Vs', 'Vm',
                    'No_mol_asu', 'MW_asu', 'No_atom_asu']]

    self.X_metrix = self.X_metrix.fillna(0)
    
    with open(os.path.join(self.output_dir,
              'recursive_feature_elimination.txt'), 'a') as text_file:
      text_file.write('Created dataframe X_metrix \n')
      text_file.write('with columns: \n')
      text_file.write(str(self.X_metrix.columns)+ '\n')
          
###############################################################################
#
#  creating training and test set
#
###############################################################################

  def split_data(self):
    '''Function which splits the input data into training set and test set.
    ******
    Input: a dataframe that contains the features and labels in columns and the samples
          in rows
    Output: sets of training and test data with an 80/20 split; X_train, X_test, y_train,
            y_test
    '''
    print('*' *80)
    print('*    Splitting data into test and training set with test=20%')
    print('*' *80)

    y = self.metrix['MR_success']

#stratified split of samples
    X_metrix_train, X_metrix_test, y_train, y_test = train_test_split(self.X_metrix, y, test_size=0.2, random_state=42, stratify=y)
    
    assert self.X_metrix.columns.all() == X_metrix_train.columns.all()

    self.X_metrix_train = X_metrix_train
    self.X_metrix_test = X_metrix_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.output_dir,
              'recursive_feature_elimination.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('X_metrix: X_metrix_train, X_metrix_test \n')
      text_file.write('y(MR_success): y_train, y_test \n')


###############################################################################
    
    #standardise data
    X_metrix_train_std = StandardScaler().fit_transform(self.X_metrix_train)
    self.X_metrix_train_std = X_metrix_train_std

###############################################################################
    
  def run_rfecv(self):  
    def run_rfecv_and_plot(X_train, y):
      print('*' *80)
      print('*    Running recursive feature elimination')
      print('*' *80)

      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      svc = SVC(kernel="linear", C=0.1, class_weight='balanced')
      # The "accuracy" scoring is proportional to the number of correct
      # classifications
      rfecv = RFECV(estimator=svc,
                    step=1,
                    min_features_to_select=5,
                    cv=StratifiedKFold(3),
                    scoring='accuracy')
      rfecv.fit(X_train, y)
      
      feature_names = ['IoverSigma', 'completeness', 'RmergeI',
                    'lowreslimit', 'RpimI', 'multiplicity', 'RmeasdiffI',
                    'wilsonbfactor', 'RmeasI', 'highreslimit', 'RpimdiffI', 
                    'RmergediffI', 'totalobservations', 'cchalf', 'totalunique',
                    'mr_reso', 'eLLG', 'tncs', 'seq_ident', 'model_res',
                    'No_atom_chain', 'MW_chain', 'No_res_chain', 'No_res_asu',
                    'likely_sg_no', 'xia2_cell_volume', 'Vs', 'Vm',
                    'No_mol_asu', 'MW_asu', 'No_atom_asu']
      
      feature_selection = rfecv.support_
      with open(os.path.join(self.output_dir,
                'recursive_feature_elimination.txt'), 'a') as text_file:
        text_file.write('Feature mask : %s \n' % feature_selection)     
      #print(feature_selection)
      
      feature_ranking = rfecv.ranking_
      with open(os.path.join(self.output_dir,
                'recursive_feature_elimination.txt'), 'a') as text_file:
        text_file.write('Feature ranking : %s \n' % feature_ranking)      
        text_file.write('Features sorted by their rank: \n')
        text_file.write(str(sorted(zip(map(
                      lambda x: round(x, 4), feature_ranking), feature_names))))
     
      
      #print(feature_ranking)
      
      print("Features sorted by their rank:")
      print(sorted(zip(map(
                       lambda x: round(x, 4), feature_ranking), feature_names)))
            
      print("Optimal number of features : %d" % rfecv.n_features_)
      
      with open(os.path.join(self.output_dir,
                'recursive_feature_elimination.txt'), 'a') as text_file:
        text_file.write('Optimal number of features : %d \n' % rfecv.n_features_)

      # Plot number of features VS. cross-validation scores
      plt.figure()
      plt.xlabel("Number of features selected")
      plt.ylabel("Cross validation score (nb of correct classifications)")
      plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
      plt.savefig(os.path.join(self.output_dir,
                             'num_features_to_use_'+datestring+'.png'), dpi=600)     
      plt.close()
      
    run_rfecv_and_plot(self.X_metrix_train_std,
                       self.y_train)
      

def run():
  args = parse_command_line()
  
  
###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)
  
  output_dir = make_output_folder(args.outdir)

###############################################################################

  feature_decomposition = RecursiveFeatureElimination(metrix, output_dir)
      


