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
  name = os.path.join(outdir, 'feature_analysis_plotting')
  os.makedirs(name, exist_ok=True)
  return name

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

  def __init__(self, data, feature_analysis_plotting):
    self.data = data
    self.feature_analysis_plotting = feature_analysis_plotting
    self.prepare_metrix_data()
    self.split_data()
    self.create_itter()
    self.plot_hist()
    self.plot_ecdf()
    self.plot_density()
   
  ###############################################################################
  #
  #  creating 3 data frames specific to the three development milestones I had
  #  1--> directly from data processing
  #  2--> after adding protein information
  #  3--> carrying out some further column transformations
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

    #database plus manually added data
#    attr_newdata_initial = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
#                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
#                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
#                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                      'diffF', 'f', 'wavelength', 'sg_number', 'cell_a', 'cell_b', 'cell_c',
#                      'cell_alpha', 'cell_beta', 'cell_gamma', 'Vcell', 'solvent_content',
#                      'Matth_coeff', 'No_atom_chain', 'No_mol_ASU',
#                      'MW_chain', 'sites_ASU']

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
#                      'volume_wilsonB_highres', 'IoverSigma_MW_ASU']
                      
#    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
#                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
#                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
#                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                      'diffF', 'f', 'wavelength',
#                      'sg_number', 'cell_a', 'cell_b', 'cell_c', 'cell_alpha',
#                      'cell_beta', 'cell_gamma','Vcell', 'solvent_content',
#                      'Vcell_Vm<Ma>', 'Matth_coeff', 'MW_ASU_sites_ASU_solvent_content',
#                      'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU', 'sites_ASU',
#                      'MW_ASU_sites_ASU', 'MW_chain_No_atom_chain', 'bragg',
#                      'volume_wilsonB_highres', 'IoverSigma_MW_ASU']

#    data_initial = self.data[attr_newdata_initial]
#    self.X_data_initial = data_initial

#    data_transform = data_initial.copy()

#    with open(os.path.join(self.feature_analysis_plotting, 'feature_analysis_plotting.txt'), 'a') #as text_file:
#      text_file.write('Preparing input data as data_initial with following attributes %s \n' %(attr_newdata_initial))

    #column transformation
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

    self.X_data_transform = self.data[["no_res", "no_frag", "longest_frag", "res_frag_ratio", "mapCC"]]

    self.X_data_transform = self.X_data_transform.fillna(0)
    
    with open(os.path.join(self.feature_analysis_plotting, 'feature_analysis_plotting.txt'), 'a') as text_file:
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
    print(y.shape)
    print(y)
    
#normal split of samples    
#    X_transform_train, X_transform_test, y_train, y_test = train_test_split(self.X_transform, y, test_size=0.2, random_state=42)

#stratified split of samples
    X_data_transform_train, X_data_transform_test, y_train, y_test = train_test_split(self.X_data_transform, y, test_size=0.2, random_state=42)#test_size=0.2,stratify=y
    
    assert self.X_data_transform.columns.all() == X_data_transform_train.columns.all()

    self.X_data_transform_train = X_data_transform_train
    self.X_data_transform_test = X_data_transform_test
    self.y_train = y_train
    self.y_test = y_test

################################################################################
#
#  creating a global feature list for iteration
#
################################################################################

  def create_itter(self):
    itter = self.X_data_transform_train.columns
    self.itter = itter
    print(self.itter)

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
      print(name)
      hist = plt.hist(self.X_data_transform_train[name], bins=365)#20
      hist = plt.xlabel(name)
      hist = plt.ylabel('number of counts')
      plt.savefig(os.path.join(self.feature_analysis_plotting, 'histogram_feature_'+name+'_'+datestring+'.png'), dpi=600)
      plt.close()
      with open(os.path.join(self.feature_analysis_plotting, 'feature_analysis_plotting.txt'), 'a') as text_file:
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
      print(name)
      '''Compute ECDF for a one-dimensional array of measurements'''
      with open(os.path.join(self.feature_analysis_plotting, 'feature_analysis_plotting.txt'), 'a') as text_file:
        text_file.write('Drawing ECDF for feature %s \n' %name)
      #Number of data points:n
      n = len(self.X_data_transform[name])
      #x-data for the ECDF:x
      x = np.sort(self.X_data_transform[name])
      #y-data for the ECDF;y
      y = np.arange(1, n+1)/n
      mean = np.mean(self.X_data_transform[name])
      median = np.median(self.X_data_transform[name])
      std = np.std(self.X_data_transform[name])
      percentile = np.percentile(self.X_data_transform[name], [25, 50, 75])
      info = self.X_data_transform[name].describe()
      samples = np.random.normal(mean, std, size=10000)
      n_theor = len(samples)
      x_theor = np.sort(samples)
      y_theor = np.arange(1, n_theor+1)/n_theor

     
      with open(os.path.join(self.feature_analysis_plotting, 'feature_analysis_plotting.txt'), 'a') as text_file:
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
      plt.savefig(os.path.join(self.feature_analysis_plotting, 'ECDF_feature_'+name+'_'+datestring+'.png'), dpi=600)
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
      with open(os.path.join(self.feature_analysis_plotting, 'feature_analysis_plotting.txt'), 'a') as text_file:
        text_file.write('Drawing ECDF for feature %s and its calculated theoretical curve\n' %name)
      mean = np.mean(self.X_data_transform[name])
      median = np.median(self.X_data_transform[name])      
      sns.distplot(self.X_data_transform[name], hist = False, kde = True,
                 kde_kws = {'linewidth': 3})  
      plt.axvline(mean, label='Mean', color='r', linestyle='--')  
      plt.axvline(median, label='Median', color='g', linestyle='--')        
      plt.legend(loc='best')
      plt.xlabel(name)
      plt.ylabel('Density')
      plt.savefig(os.path.join(self.feature_analysis_plotting, 'Density_feature_'+name+'_'+datestring+'.png'), dpi=600)
      plt.close()

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  data = load_data(args.input)

  feature_analysis_plotting = make_output_folder(args.outdir)

  ###############################################################################

  feature_analysis_plotting = FeatureAnalysisPlotting(data, feature_analysis_plotting)

