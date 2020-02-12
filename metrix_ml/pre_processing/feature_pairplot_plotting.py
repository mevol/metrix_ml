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
  name = os.path.join(outdir, 'feature_pair_plot')
  os.makedirs(name, exist_ok=True)
  return name

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

  def __init__(self, data, feature_pair_plot):
    self.data = data
    self.feature_pair_plot = feature_pair_plot
    self.prepare_metrix_data()
    self.split_data()
    self.plot_pair()
   
  ###############################################################################
  #
  #  creating data frame with column transformations
  #
  ###############################################################################

  def prepare_metrix_data(self):
    '''Function to create smaller dataframes for directly after dataprocessing, after
       adding some protein information and after carrying out some custom solumn
       transformations.
    ******
    Input: large data frame
    Output: smaller dataframes; database, man_add, transform
    '''
    print('*' *80)
    print('*    Preparing input dataframe')
    print('*' *80)

#    #database plus manually added data
#    attr_newdata_initial = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
#                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
#                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
#                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                      'diffF', 'f', 'wavelength', 'sg_number', 'cell_a', 'cell_b', 'cell_c',
#                      'cell_alpha', 'cell_beta', 'cell_gamma', 'Vcell', 'solvent_content',
#                      'Matth_coeff', 'No_atom_chain', 'No_mol_ASU',
#                      'MW_chain', 'sites_ASU', 'EP_success']

#    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
#                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
#                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
#                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                      'diffF', 'f', 'wavelength', 'wavelength**3', 'wavelength**3_Vcell',
#                      'sg_number', 'cell_a', 'cell_b', 'cell_c', 'cell_alpha',
#                      'cell_beta', 'cell_gamma','Vcell', 'solvent_content',
#                      'Vcell_Vm<Ma>', 'Matth_coeff', 'MW_ASU_sites_ASU_solvent_content',
#                      'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU', 'sites_ASU',
#                      'MW_ASU_sites_ASU', 'MW_chain_No_atom_chain', 'wilson', 'bragg',
#                      'volume_wilsonB_highres', 'IoverSigma_MW_ASU', 'EP_success']
                      
#    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
#                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
#                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
#                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                      'diffF', 'f',
#                      'sg_number', 'cell_a', 'cell_b', 'cell_c', 'cell_alpha',
#                      'cell_beta', 'cell_gamma','Vcell', 'solvent_content',
#                      'Vcell_Vm<Ma>', 'Matth_coeff', 'MW_ASU_sites_ASU_solvent_content',
#                      'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU', 'sites_ASU',
#                      'MW_ASU_sites_ASU', 'MW_chain_No_atom_chain', 'bragg',
#                      'volume_wilsonB_highres', 'IoverSigma_MW_ASU', 'EP_success']

#    data_initial = self.data[attr_newdata_initial]
#    self.X_data_initial = data_initial

#    data_transform = data_initial.copy()

#    with open(os.path.join(self.feature_pair_plot, 'feature_pair_plot.txt'), 'a') as text_file:
#      text_file.write('Preparing input data as data_initial with following attributes %s \n' %(attr_newdata_initial))

#    #column transformation
#    #MW_ASU
#    data_transform['MW_ASU'] = data_transform['MW_chain'] * data_transform['No_mol_ASU']
#
#    #MW_ASU/sites_ASU
#    data_transform['MW_ASU_sites_ASU'] = data_transform['MW_ASU'] / data_transform['sites_ASU']
#    
#    #IoverSigma/MW_ASU
#    data_transform['IoverSigma_MW_ASU'] = data_transform['IoverSigma'] / data_transform['MW_ASU']
#
#    #MW_chain/No_atom_chain
#    data_transform['MW_chain_No_atom_chain'] = data_transform['MW_chain'] / data_transform['No_atom_chain']
#
#    #MW_ASU/sites_ASU/solvent_content
#    data_transform['MW_ASU_sites_ASU_solvent_content'] = data_transform['MW_ASU_sites_ASU'] / data_transform['solvent_content']
#
#    #wavelength**3
#    data_transform['wavelength**3'] = data_transform['wavelength'] ** 3
#
#    #wavelenght**3/Vcell
#    data_transform['wavelength**3_Vcell'] = data_transform['wavelength**3'] / data_transform['Vcell']
#
#    #Vcell/Vm<Ma>
#    data_transform['Vcell_Vm<Ma>'] = data_transform['Vcell'] / (data_transform['Matth_coeff'] * data_transform['MW_chain_No_atom_chain'])
#
#    #wilson
#    data_transform['wilson'] = -2 * data_transform['wilsonbfactor']
#
#    #bragg
#    data_transform['bragg'] = (1 / data_transform['highreslimit'])**2
#
#    #use np.exp to work with series object
#    data_transform['volume_wilsonB_highres'] = data_transform['Vcell_Vm<Ma>'] * np.exp(data_transform['wilson'] * data_transform['bragg'])
    
#    self.X_data_transform = data_transform
    
#    self.X_data_transform = self.X_data_transform.fillna(0)
    
#    self.best_features = self.X_data_transform[['lowreslimit', 'anomalousslope', 'anomalousCC', 'diffI', 'diffF', 'f', 'EP_success']]

    self.X_data_transform = self.data[["no_res", "no_frag", "longest_frag", "res_frag_ratio", "mapCC", "EP_success"]]

    self.X_data_transform = self.X_data_transform.fillna(0)

    
    with open(os.path.join(self.feature_pair_plot, 'feature_pair_plot.txt'), 'a') as text_file:
      text_file.write('Created the following dataframe: data_transform \n')
      text_file.write(str(self.X_data_transform.columns)+'\n')    

    ###############################################################################
    #
    #  creating training and test set for each of the 3 dataframes
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

    y = self.data['EP_success']
    
#normal split of samples    
#    X_transform_train, X_transform_test, y_train, y_test = train_test_split(self.X_transform, y, test_size=0.2, random_state=42)

#stratified split of samples
    X_data_transform_train, X_data_transform_test, y_train, y_test = train_test_split(self.X_data_transform, y, test_size=0.2, random_state=42)#stratify=y
    
    assert self.X_data_transform.columns.all() == X_data_transform_train.columns.all()

    self.X_data_transform_train = X_data_transform_train
    self.X_data_transform_test = X_data_transform_test
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
    #graph = sns.pairplot(self.best_features, hue='EP_success')
    graph = sns.pairplot(self.X_data_transform_train, hue='EP_success')
    graph.map(corrfunc)
    handles = graph._legend_data.values()
    labels = graph._legend_data.keys()
    graph.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=1)
    graph.fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(self.feature_pair_plot, 'pairplot_'+datestring+'.png'))
      
def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  data = load_data(args.input)

  feature_pair_plot = make_output_folder(args.outdir)

  ###############################################################################

  feature_pair_plot = FeaturePairPlot(data, feature_pair_plot)

