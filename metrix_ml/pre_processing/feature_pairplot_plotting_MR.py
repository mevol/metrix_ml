# set up environment
# define command line parameters
# define location of input data
# create output directories
# start the class FeatureCorrelations

import argparse
import os
import csv
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import pearsonr#, betai
from sklearn.model_selection import train_test_split

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='various plots for feature analysis')
  
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

def load_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path, na_filter=False, skipinitialspace=True, thousands=',')

def make_output_folder(outdir):
  output_dir = os.path.join(outdir, 'feature_pair_plot')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class to analyse correlations between features
#
###############################################################################


class FeaturePairPlot(object):
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
    self.plot_pair()
   
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
                    'No_mol_asu', 'MW_asu', 'No_atom_asu', 'MR_success']]

    self.X_metrix = self.X_metrix.fillna(0)

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

################################################################################
#
#  plotting feature pairs
#
################################################################################

  def plot_pair(self):
    '''Plot a histogram for each feature.'''
    print('*' *80)
    print('*    Plotting feature pairs')
    print('*' *80)
    
    def corrfunc(x, y, **kws):
      (r, p) = pearsonr(x, y)
      #print(r, p)
      ax = plt.gca()
      #use commented-out lines below if all features used;
#     ax.annotate("r = {:.2f} ".format(r),
#                  xy=(.1, .9), xycoords=ax)
#     ax.annotate("p = {:.3f}".format(p),
#                 xy=(.1, .8), xycoords=ax)               
    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    #graph = sns.pairplot(self.best_features, hue='MR_success')
    graph = sns.pairplot(self.X_metrix_train, hue='MR_success')
    graph.map(corrfunc)
    handles = graph._legend_data.values()
    labels = graph._legend_data.keys()
    graph.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=1)
    graph.fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(self.output_dir, 'pairplot_'+datestring+'.png'))
    plt.close()
    
def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_data(args.input)

  output_dir = make_output_folder(args.outdir)

  ###############################################################################

  feature_pair_plot = FeaturePairPlot(metrix, output_dir)

