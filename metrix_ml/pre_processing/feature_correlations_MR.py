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
from pandas.plotting import scatter_matrix
from datetime import datetime
from scipy.stats import pearsonr#, betai

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colorbar as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split

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
  return pd.read_csv(csv_path)

def make_output_folder(outdir):
  output_dir = os.path.join(outdir, 'feature_correlations')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class to analyse correlations between features
#
###############################################################################


class FeatureCorrelations(object):
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
    self.plotting()
   
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
#    self.X_metrix = self.metrix[['IoverSigma', 'completeness', 'RmergeI',
#                    'lowreslimit', 'RpimI', 'multiplicity', 'RmeasdiffI',
#                    'wilsonbfactor', 'RmeasI', 'highreslimit', 'RpimdiffI', 
#                    'RmergediffI', 'totalobservations', 'cchalf', 'totalunique',
#                    'mr_reso', 'eLLG', 'tncs', 'seq_ident', 'model_res',
#                    'No_atom_chain', 'MW_chain', 'No_res_chain', 'No_res_asu',
#                    'likely_sg_no', 'xia2_cell_volume', 'Vs', 'Vm',
#                    'No_mol_asu', 'MW_asu', 'No_atom_asu']]

    self.X_metrix = self.metrix[["no_res", "no_frag", "longest_frag", "res_frag_ratio", "mapCC", "EP_success"]]

                    
    self.X_metrix = self.X_metrix.fillna(0)

    with open(os.path.join(self.output_dir,
              'feature_correlations.txt'), 'a') as text_file:
      text_file.write('Created dataframe X_metrix \n')
      text_file.write('with columns: \n')
      text_file.write(str(self.X_metrix.columns)+ '\n')

###############################################################################
#
#  creating training and test set 
#
###############################################################################

#  def split_data(self):
#    '''Function which splits the input data into training set and test set.
#    ******
#    Input: a dataframe that contains the features and labels in columns and the samples
#          in rows
#    Output: sets of training and test data with an 80/20 split; X_train, X_test, y_train,
#            y_test
#    '''
#    print('*' *80)
#    print('*    Splitting data into test and training set with test=20%')
#    print('*' *80)

#    y = self.metrix['MR_success']
#    y = self.metrix['EP_success']

#stratified split of samples
#    X_metrix_train, X_metrix_test, y_train, y_test = train_test_split(self.X_metrix, y, test_size=0.2, random_state=42, stratify=y)
    
#    assert self.X_metrix.columns.all() == X_metrix_train.columns.all()

#    self.X_metrix_train = X_metrix_train
#    self.X_metrix_test = X_metrix_test
#    self.y_train = y_train
#    self.y_test = y_test

#    with open(os.path.join(self.output_dir,
#              'feature_correlations.txt'), 'a') as text_file:
#      text_file.write('Spliting into training and test set 80-20 \n')
#      text_file.write('X_metrix: X_metrix_train, X_metrix_test \n')
#      text_file.write('y(MR_success): y_train, y_test \n')
#      text_file.write('y(EP_success): y_train, y_test \n')

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

    y = self.X_metrix['EP_success']
    
#normal split of samples    
#    X_transform_train, X_transform_test, y_train, y_test = train_test_split(self.X_transform, y, test_size=0.2, random_state=42)

#stratified split of samples
    X_metrix_train, X_metrix_test, y_train, y_test = train_test_split(self.X_metrix, y, test_size=0.2, random_state=42, stratify=y)
    
    assert self.X_metrix.columns.all() == X_metrix_train.columns.all()

    self.X_metrix_train = X_metrix_train
    self.X_metrix_test = X_metrix_test
    self.y_train = y_train
    self.y_test = y_test



###############################################################################
#
#  creating training and test set for each of the 3 dataframes
#
###############################################################################
      
  def plotting(self):      
    def calculate_pearson_cc(X_train, directory): 
      '''This functions calculates a simple statistics of
      Pearson correlation coefficient'''
      
      print('*' *80)
      print('*    Calculating Pearson Correlation Coefficient')
      print('*' *80)      
      
      attr = list(X_train.columns)
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      for a in attr:
        corr_X_train = X_train.corr()
        with open(os.path.join(directory,
               'linear_PearsonCC_values_'+datestring+'.txt'), 'a') as text_file:
               #corr_X_train[a].to_csv(text_file, header=True)
          corr_X_train[a].sort_values(ascending=False).to_csv(text_file, header=True)
        text_file.close()
        
    calculate_pearson_cc(self.X_metrix_train,
                         self.output_dir)
    
    def correlation_pvalue(X_train, directory):
    
      print('*' *80)
      print('*    Calculating Pearson Correlation Coefficient with p-values')
      print('*' *80)      
                   
      def corrcoef_loop(X_train):
        attr = list(X_train.columns)
        rows = len(attr)
        r = np.ones(shape=(rows, rows))
        p = np.ones(shape=(rows, rows))
        for i in range(rows):
          for j in range(i+1, rows):
            c1 = X_train[attr[i]]
            c2 = X_train[attr[j]]
            r_, p_ = pearsonr(c1, c2)
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
        r_squared = r**2
        self.r_squared = r_squared
        self.r = r
        self.p = p
        return self.r, self.p, self.r_squared
            
      corrcoef_loop(X_train)

      def write_corr(r, p, r_squared, X_train, directory):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        with open(os.path.join(directory,
                    'linear_corr_pvalues_'+datestring+'.csv'), 'a') as csv_file:
          attr = list(X_train)
          rows = len(attr)
          for i in range(rows):
            for j in range(i+1, rows):
              csv_file.write('%s, %s, %f, %f, %f\n' %(attr[i], attr[j], r[i,j], p[i,j], r_squared[i,j]))
        csv_file.close()
    
      write_corr(self.r,
                 self.p,
                 self.r_squared,
                 X_train,
                 directory)
  
    correlation_pvalue(self.X_metrix_train,
                       self.output_dir)
    
    def plot_scatter_matrix(X_train, directory):
      '''A function to create a scatter matrix of the data'''
      
      print('*' *80)
      print('*    Plotting Scatter Matrix for Features')
      print('*' *80)      
      
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      columns = [X_train.columns]
      for c in columns:
        if X_train.isnull().any().any() == True:
          X_train = X_train.dropna(axis=1)
      attr = list(X_train)    
      scatter_matrix(X_train[attr], figsize=(20,20))
      plt.savefig(os.path.join(directory,
                  'linear_PearsonCC_scattermatrix_'+datestring+'.png'))
      plt.close()
  
    plot_scatter_matrix(self.X_metrix_train,
                        self.output_dir)
  
    def feature_conf_mat(X_train, directory):
    
      print('*' *80)
      print('*    Plotting Confusion Matrix for Correlation Coefficients between Features')
      print('*' *80)          
    
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      corr = X_train.corr()
      
#      label_map = {
#        'IoverSigma' : '$I/\sigma$', 
#        'cchalf' : "$CC_{1/2}$", 
#        'RmergeI' : '$R_{merge}I$',
#        'RmergediffI' : '$R_{merge}(I+/I-)$', 
#        'RmeasI' : '$R_{meas}I$',
#        'RmeasdiffI' : '$R_{meas}(I+/I-)$',
#        'RpimI' : '$R_{p.i.m.}I$',
#        'RpimdiffI' : '$R_{p.i.m.}(I+/I-)$',
#        'totalobservations' : '$N_{obs total}$',
#        'totalunique' : '$N_{obs unique}$',
#        'multiplicity' : 'M',
#        'completeness' : 'T',
#        'lowreslimit' : '$d_{max}$',
#        'highreslimit' : '$d_{min}$',
#        'wilsonbfactor' : 'B',
#        'anomalousCC' : "$CC_{anom}$",
#        'diffI' : '$\Delta I/\sigma I$',
#        'diffF' : '$\Delta F/F$',
#        'anomalousslope' : '$m_{anom}$',
#        'anomalousmulti' : '$M_{anom}$',
#        'anomalouscompl' : '$T_{anom}$',
#        'xia2_cell_volume' : '$V_{cell}$',
#        'likely_sg_no' : '$N_{sg}$',
#        'Vs' : '$V_{S}$',
#        'Vm' : '$V_{M}$',
#        'MW_chain' : '$MW_{chain}$',
#        'No_atom_chain' : '$N_{atomchain}$',
#        'No_mol_asu' : '$N_{molASU}$',
#        'MW_asu' : '$MW_{ASU}$',
#        'No_res_chain' : '$N_{reschain}$',
#        'No_res_asu' : '$N_{resasu}$',
#        'No_atom_asu' : '$N_{atomasu}$',
#        'TFZ' : '$TFZ$',
#        'LLG' : '$LLG$',
#        'PAK' : '$PAK$',
#        'RMSD' : '$RMSD$',
#        'VRMS' : '$VRMS$',
#        'eLLG' : '$eLLG$',
#        'tncs' : '$TNCS$',
#        'mr_reso' : '$d_{minMR}$',
        #'mr_sg_no' : '$N_{sg_MR}$',
#        'seq_ident' : '$i$',
#        'model_res' : '$d_{minmodel}$'
#        }

#      yticklabels = [label_map[key] for key in corr.columns]
#      xticklabels = [label_map[key] for key in corr.columns]
      
      fig = plt.figure(figsize=(20, 20))

      ax = plt.gca()
      im = ax.imshow(corr, cmap=sns.diverging_palette(0, 255, sep=32, n=256,
                                                  center='light', as_cmap=True))
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.2)
      cax.tick_params(labelsize=14)
      plt.colorbar(im, cax=cax).set_label("Pearson's Correlation Coefficient",
                                          fontsize=14)
      #ax.set_xticks(np.arange(len(xticklabels)))
      ax.set_xticks(np.arange(len(X_train.columns)))
      #ax.set_xticklabels(xticklabels, rotation=90, fontsize=14)
      ax.set_xticklabels(X_train.columns, rotation=90, fontsize=14)      
      #ax.set_yticks(np.arange(len(yticklabels)))
      ax.set_yticks(np.arange(len(X_train.columns)))
      #ax.set_yticklabels(yticklabels, fontsize=14)
      ax.set_yticklabels(X_train.columns, fontsize=14)
      #fig.suptitle("Linear Pearson's Correlation Coefficient", fontsize=16)
      #ax.set_title('Feature1 using Data2', fontsize=12)
      plt.tight_layout()
      
      plt.savefig(os.path.join(directory,
                  'feature_corr_matrix_'+datestring+'.png'), dpi=600)
      plt.close()
    
    feature_conf_mat(self.X_metrix_train,
                     self.output_dir)
    
def run():
  args = parse_command_line()
  
  
###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  output_dir = make_output_folder(args.outdir)

###############################################################################

  feature_correlations = FeatureCorrelations(metrix, output_dir)
   
   
   
   
