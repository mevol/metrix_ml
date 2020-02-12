# set up environment
# define command line parameters
# define location of input data
# create output directories
# start the class FeatureCorrelations

import argparse
import pandas as pd
import os
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='Correlation coefficient analysis of features')
  
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
#  load the data from CSV file and creating output directory
#
###############################################################################

def load_metrix_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path, na_filter=False, skipinitialspace=True, thousands=',')

def make_output_folder(outdir):
  output_dir = os.path.join(outdir, 'polynomial_features')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class to create polynomials of features
#
###############################################################################


class CreatePolynomialFeatures(object):
  '''A class to help analyse the data;
  try to identify linear correlations in the data;
  calculate Pearson Correlation Coefficient with and without
  p-values; create a scatter matrix; inout data must not contain
  any strings or NaN values; also remove any columns with 
  categorical data or transform them first; remove any text except column labels'''

  def __init__(self, metrix, output_dir):
    self.metrix = metrix
    self.output_dir = output_dir
    self.prepare_metrix_data()
    self.split_data()
    self.polynomials()
   
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
              'polynomial_features.txt'), 'a') as text_file:
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
              'polynomial_features.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('X_metrix: X_metrix_train, X_metrix_test \n')
      text_file.write('y(MR_success): y_train, y_test \n')
      
###############################################################################
      
  def polynomials(self):
    def get_polynomials(X_train, y_train):
    
      print('*' *80)
      print('*    Creating polynomial features of degree=2 and fitting the data')
      print('*' *80)
     
      poly = PolynomialFeatures(degree=3)
      X_train_poly = poly.fit(X_train)

      self.X_train_poly = X_train_poly
      
      num_poly = self.X_train_poly.n_output_features_
      feature_names = X_train_poly.get_feature_names(X_train.columns)

      with open(os.path.join(self.output_dir,
                'polynomial_features.txt'), 'a') as text_file:
        text_file.write('The number of polynomial features is: %s \n' %num_poly)
        text_file.write('The polynomial features names are: %s \n' %feature_names)
        
      print('*' *80)
      print('*    Transforming the data with polynomial features')
      print('*' *80)
      
      self.X_train_poly = self.X_train_poly.transform(X_train)
      
      print('*' *80)
      print('*    Run randomforest data with polynomial features')
      print('*' *80)
             
      self.forest_clf_poly = RandomForestClassifier(random_state=100,
                                                    class_weight='balanced')
      self.forest_clf_poly.fit(self.X_train_poly, self.y_train)
      
      feature_importances = self.forest_clf_poly.feature_importances_
      feature_importances_ls = sorted(zip(feature_importances,
                                          feature_names), reverse=True)
      
      with open(os.path.join(self.output_dir,
                'polynomial_features.txt'), 'a') as text_file:
        text_file.write('List of all polynomial features: %s \n' %feature_importances_ls)
        text_file.write('The 25 highest scoring polynomial features are: %s \n' %feature_importances_ls[:25])
      
      self.feature_importances_ls = feature_importances_ls
      
      def feature_importances_best_25(feature_list, directory):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        feature_list.sort(key=lambda x: x[1], reverse=True)
        feature = list(zip(*feature_list))[1]
        score = list(zip(*feature_list))[0]
        x_pos = np.arange(len(feature))
        plt.bar(x_pos, score,align='center')
        plt.xticks(x_pos, feature, rotation=90, fontsize=4)
        plt.title('Histogram of best 25 Feature Importances for RandomForest')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(directory,
             'feature_importances_bar_plot_best25_'+datestring+'.png'), dpi=600)     
        plt.close()
        
      feature_importances_best_25(self.feature_importances_ls[:25],
                                  self.output_dir)  
      
      def feature_importances_pandas(clf, X_train, feature_names, directory):   
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')      
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]

        # Plot the feature importances of the forest
        plt.figure(figsize=(20,10))
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r",
                yerr=std[indices], align="center")
        feature_names = feature_names
        plt.xticks(range(X_train.shape[1]), indices, rotation=90, fontsize=4)
        plt.xlim([-1, X_train.shape[1]])
        plt.tight_layout()
        plt.savefig(os.path.join(directory,
         'feature_importances_alltreeserr_ordered_bar_plot_'+datestring+'.png'),
         dpi=600)     
        plt.close()

      feature_importances_pandas(self.forest_clf_poly,
                                 self.X_train_poly,
                                 feature_names,
                                 self.output_dir)
    
      def feature_importances_pandas2(clf, X_train, columns, directory):   
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')      
        feature_list = []
        for tree in clf.estimators_:
          feature_importances_ls = tree.feature_importances_
          feature_list.append(feature_importances_ls)
        
        df = pd.DataFrame(feature_list, columns=columns)
        df_mean = df[columns].mean(axis=0)
        df_std = df[columns].std(axis=0)
        #df_mean.plot(kind='bar', color='b', yerr=[df_std], align="center",
        #figsize=(20,10), title="Feature importances", rot=60)
        df_mean.plot(kind='bar', color='b', yerr=[df_std], align="center",
                     figsize=(20,10), rot=90, fontsize=4)
        plt.title('Histogram of Feature Importances over all RandomForest using features')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(directory,
        'feature_importances_alltreeserr_unordered_bar_plot_'+datestring+'.png'),
        dpi=600)
        plt.close()
      
      feature_importances_pandas2(self.forest_clf_poly,
                                  self.X_train_poly,
                                  feature_names,
                                  self.output_dir)
     
    get_polynomials(self.X_metrix_train,
                    self.y_train)      

def run():
  args = parse_command_line()
  
  
###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  output_dir = make_output_folder(args.outdir)

###############################################################################

  polynomial_features = CreatePolynomialFeatures(metrix, output_dir)

