# set up environment
# define command line parameters
# define location of input data
# create output directories
# start the class FeatureCorrelations

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import csv
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
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
  output_dir = os.path.join(outdir, 'feature_analysis_plotting')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class to analyse correlations between features
#
###############################################################################


class FeatureAnalysisPlotting(object):
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
    self.create_itter()
    self.plot_hist()
    self.plot_ecdf()
    self.plot_density()
   
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
    self.X_metrix = self.metrix[['anomalousCC', 'IoverSigma', 'completeness',
                    'diffI', 'RmergeI', 'lowreslimit', 'RpimI', 'multiplicity',
                    'RmeasdiffI', 'anomalousslope', 'diffF', 'wilsonbfactor',
                    'RmeasI', 'highreslimit', 'RpimdiffI', 'anomalousmulti',
                    'RmergediffI', 'totalobservations', 'anomalouscompl',
                    'cchalf', 'totalunique', 'LLG', 'TFZ', 'PAK', 'mr_reso',
                    'RMSD', 'VRMS', 'eLLG', 'tncs', 'seq_ident', 'model_res',
                    'No_atom_chain', 'MW_chain', 'No_res_chain', 'No_res_asu',
                    'likely_sg_no', 'xia2_cell_volume', 'Vs', 'Vm',
                    'No_mol_asu', 'MW_asu', 'No_atom_asu']]

    self.X_metrix = self.X_metrix.fillna(0)

    with open(os.path.join(self.output_dir,
              'feature_analysis_plotting.txt'), 'a') as text_file:
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
              'feature_analysis_plotting.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('X_metrix: X_metrix_train, X_metrix_test \n')
      text_file.write('y(MR_success): y_train, y_test \n')

################################################################################
#
#  creating a global feature list for iteration
#
################################################################################

  def create_itter(self):
    itter = self.X_metrix_train.columns
    self.itter = itter

################################################################################
#
#  plotting histogram for each feature
#
################################################################################

  def plot_hist(self):
    '''Plot a histogram for each feature.'''
    print('*' *80)
    print('*    Plotting histogram for each feature')
    print('*' *80)
    
    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    for name in self.itter:
      hist = plt.hist(self.X_metrix_train[name], bins=20)
      hist = plt.xlabel(name)
      hist = plt.ylabel('number of counts')
      plt.savefig(os.path.join(self.output_dir,
                      'histogram_feature_'+name+'_'+datestring+'.png'), dpi=600)
      plt.close()
      with open(os.path.join(self.output_dir,
                'feature_analysis_plotting.txt'), 'a') as text_file:
        text_file.write('Drawing histogram for feature %s \n' %name)
      
################################################################################
#
#  plotting empirical cumulative distribution function for each feature
#
################################################################################
      
  def plot_ecdf(self):
    '''Plot empirical cumulative distribution function (ECDF) for each feature.'''
    print('*' *80)
    print('*    Plotting ECDF for each feature')
    print('*' *80)

    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    
    for name in self.itter:
      '''Compute ECDF for a one-dimensional array of measurements'''
      with open(os.path.join(self.output_dir,
                'feature_analysis_plotting.txt'), 'a') as text_file:
        text_file.write('Drawing ECDF for feature %s \n' %name)
      #Number of data points:n
      n = len(self.X_metrix_train[name])
      #x-data for the ECDF:x
      x = np.sort(self.X_metrix_train[name])
      #y-data for the ECDF;y
      y = np.arange(1, n+1)/n
      mean = np.mean(self.X_metrix_train[name])
      median = np.median(self.X_metrix_train[name])
      std = np.std(self.X_metrix_train[name])
      percentile = np.percentile(self.X_metrix_train[name], [25, 50, 75])
      info = self.X_metrix_train[name].describe()
      samples = np.random.normal(mean, std, size=10000)
      n_theor = len(samples)
      x_theor = np.sort(samples)
      y_theor = np.arange(1, n_theor+1)/n_theor

     
      with open(os.path.join(self.output_dir,
                'feature_analysis_plotting.txt'), 'a') as text_file:
        text_file.write('The basic stats for feature %s are: \n' %name)
        text_file.write('mean: '+str(info[1])+'\n')
        text_file.write('median: '+str(median)+'\n')
        text_file.write('std: '+str(info[2])+'\n')
        text_file.write('min: '+str(info[3])+'\n')
        text_file.write('25%: '+str(info[4])+'\n')
        text_file.write('50%: '+str(info[5])+'\n')
        text_file.write('75%: '+str(info[6])+'\n')
        text_file.write('max: '+str(info[7])+'\n')
      plt.plot(x, y, marker='.', linestyle='none', label='observed CDF')
      plt.plot(x_theor, y_theor, label='theoretical CDF')
      plt.axvline(mean, label='Mean', color='r', linestyle='--')
      plt.axvline(median, label='Median', color='g', linestyle='--')
      plt.text(mean, 0.9, 'Mean: %.2f' %mean)
      plt.text(median, 0.8, 'Median: %.2f' %median)
      plt.xlabel(name)
      plt.ylabel('(E)CDF')
      plt.margins(0.02)
      plt.legend(loc='best')
      plt.savefig(os.path.join(self.output_dir,
                           'ECDF_feature_'+name+'_'+datestring+'.png'), dpi=600)
      plt.close()

################################################################################
#
#  plotting probability density function for each feature to evaluate continous
#  data and its cumulative distribution function
#
################################################################################

  def plot_density(self):
    '''Plot probability density function (PDF) for each feature.'''
    print('*' *80)
    print('*    Plotting density curve for each feature')
    print('*' *80)

    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    
    for name in self.itter:
      '''Compute ECDF for a one-dimensional array of measurements'''
      with open(os.path.join(self.output_dir,
                'feature_analysis_plotting.txt'), 'a') as text_file:
        text_file.write('Drawing ECDF for feature %s and its calculated theoretical curve\n' %name)
      mean = np.mean(self.X_metrix_train[name])
      median = np.median(self.X_metrix_train[name])      
      sns.distplot(self.X_metrix_train[name], hist = False, kde = True,
                 kde_kws = {'linewidth': 3})  
      plt.axvline(mean, label='Mean', color='r', linestyle='--')  
      plt.axvline(median, label='Median', color='g', linestyle='--')        
      plt.legend(loc='best')
      plt.xlabel(name)
      plt.ylabel('Density')
      plt.savefig(os.path.join(self.output_dir,
                        'Density_feature_'+name+'_'+datestring+'.png'), dpi=600)
      plt.close()

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  matrix = load_data(args.input)

  output_dir = make_output_folder(args.outdir)

  ###############################################################################

  feature_analysis_plotting = FeatureAnalysisPlotting(matrix, output_dir)

