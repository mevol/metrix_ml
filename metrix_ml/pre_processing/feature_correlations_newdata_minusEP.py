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
  names = ['newdata_minusEP', 'bbbb']
  result = []
  for name in names:
    name = os.path.join(outdir, 'feature_correlations', name)
    os.makedirs(name, exist_ok=True)
    result.append(name)
  return result

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

  def __init__(self, metrix, newdata_minusEP, bbbb):
    self.metrix=metrix
    self.newdata_minusEP=newdata_minusEP
    self.prepare_metrix_data()
    self.split_data()
    self.plotting()
   
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
    print('*    Preparing input dataframes metrix_newdata_minusEP')
    print('*' *80)

    #database plus manually added data
    attr_newdata_initial = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
                      'diffF', 'f','wavelength', 'sg_number', 'cell_a', 'cell_b', 'cell_c',
                      'cell_alpha', 'cell_beta', 'cell_gamma', 'Vcell', 'solvent_content',
                      'Matth_coeff', 'No_atom_chain', 'No_mol_ASU',
                      'MW_chain', 'sites_ASU']
                      
#    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergeI', 'RmergediffI', 'RmeasI',
#                      'RmeasdiffI', 'RpimI', 'RpimdiffI', 'totalobservations',
#                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
#                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                      'diffF', 'f','wavelength', 'wavelength**3', 'wavelength**3/Vcell',
#                      'sg_number', 'cell_a', 'cell_b', 'cell_c', 'cell_alpha',
#                      'cell_beta', 'cell_gamma','Vcell', 'solvent_content',
#                      'Vcell/Vm<Ma>', 'Matth_coeff', 'MW_ASU/sites_ASU/solvent_content',
#                      'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU', 'sites_ASU',
#                      'MW_ASU/sites_ASU', 'MW_chain/No_atom_chain', 'wilson', 'bragg',
#                      'volume_wilsonB_highres', 'IoverSigma/MW_ASU']
                      
    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergeI', 'RmergediffI', 'RmeasI',
                      'RmeasdiffI', 'RpimI', 'RpimdiffI', 'totalobservations',
                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
                      'diffF', 'f', 'wavelength',
                      'sg_number', 'cell_a', 'cell_b', 'cell_c', 'cell_alpha',
                      'cell_beta', 'cell_gamma','Vcell', 'solvent_content',
                      'Vcell/Vm<Ma>', 'Matth_coeff', 'MW_ASU/sites_ASU/solvent_content',
                      'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU', 'sites_ASU',
                      'MW_ASU/sites_ASU', 'MW_chain/No_atom_chain', 'bragg',
                      'volume_wilsonB_highres', 'IoverSigma/MW_ASU']


    metrix_newdata_initial = self.metrix[attr_newdata_initial]

    metrix_newdata_transform = metrix_newdata_initial.copy()

    with open(os.path.join(self.newdata_minusEP, 'feature_correlations.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_newdata_initial with following attributes %s \n' %(attr_newdata_initial))

    #column transformation
    #MW_ASU
    metrix_newdata_transform['MW_ASU'] = metrix_newdata_transform['MW_chain'] * metrix_newdata_transform['No_mol_ASU']

    #MW_ASU/sites_ASU
    metrix_newdata_transform['MW_ASU/sites_ASU'] = metrix_newdata_transform['MW_ASU'] / metrix_newdata_transform['sites_ASU']
    
    #IoverSigma/MW_ASU
    metrix_newdata_transform['IoverSigma/MW_ASU'] = metrix_newdata_transform['IoverSigma'] / metrix_newdata_transform['MW_ASU']

    #MW_chain/No_atom_chain
    metrix_newdata_transform['MW_chain/No_atom_chain'] = metrix_newdata_transform['MW_chain'] / metrix_newdata_transform['No_atom_chain']

    #MW_ASU/sites_ASU/solvent_content
    metrix_newdata_transform['MW_ASU/sites_ASU/solvent_content'] = metrix_newdata_transform['MW_ASU/sites_ASU'] / metrix_newdata_transform['solvent_content']

#    #wavelength**3
#    metrix_newdata_transform['wavelength**3'] = metrix_newdata_transform['wavelength'] ** 3

#    #wavelenght**3/Vcell
#    metrix_newdata_transform['wavelength**3/Vcell'] = metrix_newdata_transform['wavelength**3'] / metrix_newdata_transform['Vcell']

    #Vcell/Vm<Ma>
    metrix_newdata_transform['Vcell/Vm<Ma>'] = metrix_newdata_transform['Vcell'] / (metrix_newdata_transform['Matth_coeff'] * metrix_newdata_transform['MW_chain/No_atom_chain'])

    #wilson
    metrix_newdata_transform['wilson'] = -2 * metrix_newdata_transform['wilsonbfactor']

    #bragg
    metrix_newdata_transform['bragg'] = (1 / metrix_newdata_transform['highreslimit'])**2

    #use np.exp to work with series object
    metrix_newdata_transform['volume_wilsonB_highres'] = metrix_newdata_transform['Vcell/Vm<Ma>'] * np.exp(metrix_newdata_transform['wilson'] * metrix_newdata_transform['bragg'])
    
    self.X_newdata_transform = metrix_newdata_transform

    with open(os.path.join(self.newdata_minusEP, 'feature_correlations.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_newdata_transform \n')
      text_file.write(str(self.X_newdata_transform.columns)+'\n')    

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

    y = self.metrix['EP_success']
    
#normal split of samples    
#    X_transform_train, X_transform_test, y_train, y_test = train_test_split(self.X_transform, y, test_size=0.2, random_state=42)

#stratified split of samples
    X_newdata_transform_train, X_newdata_transform_test, y_train, y_test = train_test_split(self.X_newdata_transform, y, test_size=0.2, random_state=42, stratify=y)
    
    assert self.X_newdata_transform.columns.all() == X_newdata_transform_train.columns.all()

    self.X_newdata_transform_train = X_newdata_transform_train
    self.X_newdata_transform_test = X_newdata_transform_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.newdata_minusEP, 'feature_correlations.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_newdata_transform: X_newdata_transform_train, X_newdata_transform_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')

    print('*' *80)
    print('*    Ordering columns')
    print('*' *80)

#    self.X_newdata_transform_train_ordered = self.X_newdata_transform_train[
#                         ['IoverSigma', 'cchalf', 'RmergeI', 'RmergediffI', 'RmeasI',
#                          'RmeasdiffI', 'RpimI', 'RpimdiffI', 'totalobservations',
#                          'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                          'highreslimit', 'wilsonbfactor', 'wilson', 'bragg', 'anomalousslope',
#                          'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                          'diffF', 'f','wavelength', 'wavelength**3', 'wavelength**3/Vcell',
#                          'Vcell', 'sg_number', 'cell_a', 'cell_b', 'cell_c',
#                          'cell_alpha', 'cell_beta', 'cell_gamma','solvent_content',
#                          'Vcell/Vm<Ma>', 'volume_wilsonB_highres', 'Matth_coeff', 'MW_ASU/sites_ASU/solvent_content',
#                          'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU',
#                          'sites_ASU', 'MW_ASU/sites_ASU', 'IoverSigma/MW_ASU',
#                          'MW_chain/No_atom_chain'
#                          ]]                              

    self.X_newdata_transform_train_ordered = self.X_newdata_transform_train[
                         ['IoverSigma', 'cchalf', 'RmergeI', 'RmergediffI', 'RmeasI',
                          'RmeasdiffI', 'RpimI', 'RpimdiffI', 'totalobservations',
                          'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
                          'highreslimit', 'wilsonbfactor', 'bragg', 'anomalousslope',
                          'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
                          'diffF', 'f', 'wavelength',
                          'Vcell', 'sg_number', 'cell_a', 'cell_b', 'cell_c',
                          'cell_alpha', 'cell_beta', 'cell_gamma','solvent_content',
                          'Vcell/Vm<Ma>', 'volume_wilsonB_highres', 'Matth_coeff', 'MW_ASU/sites_ASU/solvent_content',
                          'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU',
                          'sites_ASU', 'MW_ASU/sites_ASU', 'IoverSigma/MW_ASU',
                          'MW_chain/No_atom_chain'
                          ]]                              

    ###############################################################################
    #
    #  creating training and test set for each of the 3 dataframes
    #
    ###############################################################################
      
  def plotting(self):      
    def calculate_pearson_cc(X_train, name, directory): 
      '''This functions calculates a simple statistics of
      Pearson correlation coefficient'''
      
      print('*' *80)
      print('*    Calculating Pearson Correlation Coefficient')
      print('*' *80)      
      
      attr = list(X_train)
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      for a in attr:
        corr_X_train = X_train.corr()        
        with open(os.path.join(directory, 'linear_PearsonCC_values_'+name+datestring+'.txt'), 'a') as text_file:
          corr_X_train[a].sort_values(ascending=False).to_csv(text_file)
        text_file.close()
        
    calculate_pearson_cc(self.X_newdata_transform_train_ordered, 'newdata_minusEP', self.newdata_minusEP)
    
    def correlation_pvalue(X_train, name, directory):
    
      print('*' *80)
      print('*    Calculating Pearson Correlation Coefficient with p-values')
      print('*' *80)      
                   
      def corrcoef_loop(X_train):
        attr = list(X_train)
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
        print(r)
        r_squared = r**2
        print(r_squared)
        self.r_squared = r_squared
        self.r = r
        self.p = p
        return self.r, self.p, self.r_squared
            
      corrcoef_loop(X_train)

      def write_corr(r, p, r_squared, X_train, name, directory):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        with open(os.path.join(directory,
                           'linear_corr_pvalues_'+name+datestring+'.csv'), 'a') as csv_file:
          attr = list(X_train)
          rows = len(attr)
          for i in range(rows):
            for j in range(i+1, rows):
              csv_file.write('%s, %s, %f, %f, %f\n' %(attr[i], attr[j], r[i,j], p[i,j], r_squared[i,j]))
        csv_file.close()
    
      write_corr(self.r, self.p, self.r_squared, X_train, name, directory)
  
    correlation_pvalue(self.X_newdata_transform_train_ordered, 'newdata_minusEP', self.newdata_minusEP)
    
    def plot_scatter_matrix(X_train, name, directory):
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
      plt.savefig(os.path.join(directory, 'linear_PearsonCC_scattermatrix_'+name+datestring+'.png'))
      plt.close()
  
    plot_scatter_matrix(self.X_newdata_transform_train_ordered, 'newdata_minusEP', self.newdata_minusEP)
  
    def feature_conf_mat(X_train, name, directory):
    
      print('*' *80)
      print('*    Plotting Confusion Matrix for Correlation Coefficients between Features')
      print('*' *80)          
    
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      corr = X_train.corr()
      
      label_map = {
        'IoverSigma' : '$I/\sigma$', 
        'cchalf' : "$CC_{1/2}$", 
        'RmergeI' : '$R_{merge}I$',
        'RmergediffI' : '$R_{merge}(I+/I-)$', 
        'RmeasI' : '$R_{meas}I$',
        'RmeasdiffI' : '$R_{meas}(I+/I-)$',
        'RpimI' : '$R_{p.i.m.}I$',
        'RpimdiffI' : '$R_{p.i.m.}(I+/I-)$',
        'totalobservations' : '$N_{obs total}$',
        'totalunique' : '$N_{obs unique}$',
        'multiplicity' : 'M',
        'completeness' : 'T',
        'lowreslimit' : '$d_{max}$',
        'highreslimit' : '$d_{min}$',
        'wilsonbfactor' : 'B',
        'anomalousslope' : '$m_{anom}$',
        'anomalousCC' : '$CC_{anom}$',
        'anomalousmulti' : '$M_{anom}$',
        'anomalouscompl' : '$T_{anom}$',
        'diffI' : '$\Delta I/\sigma I$',
        'diffF' : '$\Delta F/F$',
        'f' : "$f''_{theor}$",
        'wavelength' : '$\lambda$',
#        'wavelength**3' : '$\lambda ^{3}$',
#        'wavelength**3/Vcell' : '$\lambda _{V}$',
        'Vcell' : '$V_{cell}$',
        'sg_number' : '$N_{sg}$',
        'cell_a' : 'a',
        'cell_b' : 'b',
        'cell_c' : 'c',
        'cell_alpha' : r"$\alpha$",
        'cell_beta' : r"$\beta$",
        'cell_gamma' : r'$\gamma$',
        'solvent_content' : '$V_{S}$',
        'Vcell/Vm<Ma>' : '$N_{cell}$',
        'Matth_coeff' : '$V_{m}$',
        'MW_ASU/sites_ASU/solvent_content' : '$MWS_{ASU Vs}$',
        'MW_chain' : '$MW_{chain}$',
        'No_atom_chain' : '$N_{atomchain}$',
        'No_mol_ASU' : '$N_{molASU}$',
        'MW_ASU' : '$MW_{ASU}$',
        'sites_ASU' : '$N_{sitesASU}$',
        'MW_ASU/sites_ASU' : '$MWS_{ASU}$',
        'IoverSigma/MW_ASU' : '$I_{ASU}$',
        'MW_chain/No_atom_chain' : '$AV_{Z}$',
#        'wilson' : '-2B',
        'bragg' : '$d_{inv}$',
        'volume_wilsonB_highres' : 'L'}

      yticklabels = [label_map[key] for key in corr.columns]
      xticklabels = [label_map[key] for key in corr.columns]
      
      fig = plt.figure(figsize=(15, 15))

      ax = plt.gca()
      im = ax.imshow(corr, cmap=sns.diverging_palette(0, 255, sep=32, n=256, center='light', as_cmap=True))
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.2)
      cax.tick_params(labelsize=14)
      plt.colorbar(im, cax=cax).set_label("Pearson's Correlation Coefficient", fontsize=14)
      ax.set_xticks(np.arange(len(xticklabels)))
      ax.set_xticklabels(xticklabels, rotation=90, fontsize=14)
      ax.set_yticks(np.arange(len(yticklabels)))
      ax.set_yticklabels(yticklabels, fontsize=14)
      #fig.suptitle("Linear Pearson's Correlation Coefficient", fontsize=16)
      #ax.set_title('Feature1 using Data2', fontsize=12)
      plt.tight_layout()
      
      plt.savefig(os.path.join(directory, 'feature_confusion_matrix_'+name+datestring+'.png'), dpi=600)
    
    feature_conf_mat(self.X_newdata_transform_train_ordered, 'newdata_minusEP', self.newdata_minusEP)
    
def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  newdata_minusEP, bbbb= make_output_folder(args.outdir)

  ###############################################################################

  feature_correlations = FeatureCorrelations(metrix, newdata_minusEP, bbbb)
   
   
   
   
