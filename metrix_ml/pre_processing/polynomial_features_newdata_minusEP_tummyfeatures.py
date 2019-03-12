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
  names = ['newdata_minusEP', 'bbbb']
  result = []
  for name in names:
    name = os.path.join(outdir, 'polynomial_features', name)
    os.makedirs(name, exist_ok=True)
    result.append(name)
  return result

###############################################################################
#
#  class to analyse correlations between features
#
###############################################################################


class CreatePolynomialFeatures(object):
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
    self.polynomials()
   
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
    print('*    Preparing input dataframes metrix_database, metrix_man_add, metrix_transform, metrix_prot_screen_trans')
    print('*' *80)

    #database plus manually added data
    attr_newdata_initial = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
                      'diffF', 'wavelength', 'sg_number', 'cell_a', 'cell_b', 'cell_c',
                      'cell_alpha', 'cell_beta', 'cell_gamma', 'Vcell', 'solvent_content',
                      'Matth_coeff', 'No_atom_chain', 'No_mol_ASU',
                      'MW_chain', 'sites_ASU']

    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
                      'diffF', 'wavelength', 'wavelength**3', 'wavelength**3/Vcell',
                      'sg_number', 'cell_a', 'cell_b', 'cell_c', 'cell_alpha',
                      'cell_beta', 'cell_gamma','Vcell', 'solvent_content',
                      'Vcell/Vm<Ma>', 'Matth_coeff', 'MW_ASU/sites_ASU/solvent_content',
                      'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU', 'sites_ASU',
                      'MW_ASU/sites_ASU', 'MW_chain/No_atom_chain', 'wilson', 'bragg',
                      'volume_wilsonB_highres', 'IoverSigma/MW_ASU']
                      

    metrix_newdata_initial = self.metrix[attr_newdata_initial]
    self.X_newdata_initial = metrix_newdata_initial

    metrix_newdata_transform = metrix_newdata_initial.copy()

    with open(os.path.join(self.newdata_minusEP, 'polynomial_features.txt'), 'a') as text_file:
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

    #wavelength**3
    metrix_newdata_transform['wavelength**3'] = metrix_newdata_transform['wavelength'] ** 3

    #wavelenght**3/Vcell
    metrix_newdata_transform['wavelength**3/Vcell'] = metrix_newdata_transform['wavelength**3'] / metrix_newdata_transform['Vcell']

    #Vcell/Vm<Ma>
    metrix_newdata_transform['Vcell/Vm<Ma>'] = metrix_newdata_transform['Vcell'] / (metrix_newdata_transform['Matth_coeff'] * metrix_newdata_transform['MW_chain/No_atom_chain'])

    #wilson
    metrix_newdata_transform['wilson'] = -2 * metrix_newdata_transform['wilsonbfactor']

    #bragg
    metrix_newdata_transform['bragg'] = (1 / metrix_newdata_transform['highreslimit'])**2

    #use np.exp to work with series object
    metrix_newdata_transform['volume_wilsonB_highres'] = metrix_newdata_transform['Vcell/Vm<Ma>'] * np.exp(metrix_newdata_transform['wilson'] * metrix_newdata_transform['bragg'])
    
    self.X_newdata_tummy = metrix_newdata_transform[['diffI', 'anomalousCC', 'MW_ASU/sites_ASU', 'lowreslimit', 'anomalousslope', 'diffF', 'MW_ASU/sites_ASU/solvent_content',
    'Matth_coeff', 'sg_number', 'cchalf', 'anomalouscompl', 'solvent_content']]

    #self.X_newdata_transform = metrix_newdata_transform
    
    #self.X_newdata_transform.to_csv(os.path.join(self.newdata, 'transformed_dataframe.csv'))
    
    #np.isnan(self.X_newdata_transform)
    #print(np.where(np.isnan(self.X_newdata_transform)))
    #self.X_newdata_transform = np.nan_to_num(self.X_newdata_transform)
    self.X_newdata_tummy = self.X_newdata_tummy.fillna(0)
    
    with open(os.path.join(self.newdata_minusEP, 'polynomial_features.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_newdata_transform \n')
      text_file.write(str(self.X_newdata_tummy.columns)+'\n')    

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
#    X_man_add_train, X_man_add_test, y_train, y_test = train_test_split(self.X_man_add, y, test_size=0.2, random_state=42)

#stratified split of samples
    X_newdata_transform_train, X_newdata_transform_test, y_train, y_test = train_test_split(self.X_newdata_tummy, y, test_size=0.2, random_state=42, stratify=y)
    
    assert self.X_newdata_tummy.columns.all() == X_newdata_transform_train.columns.all()

    self.X_newdata_transform_train = X_newdata_transform_train
    self.X_newdata_transform_test = X_newdata_transform_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.newdata_minusEP, 'polynomial_features.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_newdata_transform: X_newdata_transform_train, X_newdata_transform_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')
      
###############################################################################
      
  def polynomials(self):
    def get_polynomials(X_train, y_train):
    
      print('*' *80)
      print('*    Creating polynomial features of degree=2 and fitting the data')
      print('*' *80)
     
      poly = PolynomialFeatures(degree=3)
      X_newdata_train_poly = poly.fit(X_train)

      self.X_newdata_train_poly = X_newdata_train_poly
      
      num_poly = self.X_newdata_train_poly.n_output_features_
      feature_names = X_newdata_train_poly.get_feature_names(X_train.columns)

      with open(os.path.join(self.newdata_minusEP, 'polynomial_features.txt'), 'a') as text_file:
        text_file.write('The number of polynomial features is: %s \n' %num_poly)
        text_file.write('The polynomial features names are: %s \n' %feature_names)
        
      print('*' *80)
      print('*    Transforming the data with polynomial features')
      print('*' *80)
  
      
      self.X_newdata_train_poly = self.X_newdata_train_poly.transform(X_train)
      
      print('*' *80)
      print('*    Run randomforest data with polynomial features')
      print('*' *80)
      
       
      self.forest_clf_poly = RandomForestClassifier(random_state=100, class_weight='balanced')
      self.forest_clf_poly.fit(self.X_newdata_train_poly, self.y_train)
      
      feature_importances = self.forest_clf_poly.feature_importances_
      feature_importances_ls = sorted(zip(feature_importances, feature_names), reverse=True)
      
      with open(os.path.join(self.newdata_minusEP, 'polynomial_features.txt'), 'a') as text_file:
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
        plt.savefig(os.path.join(directory, 'feature_importances_bar_plot_best25_'+datestring+'.png'), dpi=600)     
        plt.close()
        
      feature_importances_best_25(self.feature_importances_ls[:12], self.newdata_minusEP)  
      
      def feature_importances_pandas(clf, X_train, feature_names, directory):   
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')      
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]

        # Plot the feature importances of the forest
        plt.figure(figsize=(20,10))
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        feature_names = feature_names
        plt.xticks(range(X_train.shape[1]), indices, rotation=90, fontsize=4)
        plt.xlim([-1, X_train.shape[1]])
        plt.tight_layout()
        plt.savefig(os.path.join(directory, 'feature_importances_alltreeserr_ordered_bar_plot_'+datestring+'.png'), dpi=600)     
        plt.close()

      feature_importances_pandas(self.forest_clf_poly, self.X_newdata_train_poly, feature_names, self.newdata_minusEP)
    
      def feature_importances_pandas2(clf, X_train, columns, directory):   
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')      
        feature_list = []
        for tree in clf.estimators_:
          feature_importances_ls = tree.feature_importances_
          feature_list.append(feature_importances_ls)
        
        df = pd.DataFrame(feature_list, columns=columns)
        df_mean = df[columns].mean(axis=0)
        df_std = df[columns].std(axis=0)
        #df_mean.plot(kind='bar', color='b', yerr=[df_std], align="center", figsize=(20,10), title="Feature importances", rot=60)
        df_mean.plot(kind='bar', color='b', yerr=[df_std], align="center", figsize=(20,10), rot=90, fontsize=4)
        plt.title('Histogram of Feature Importances over all RandomForest using features')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(directory, 'feature_importances_alltreeserr_unordered_bar_plot_'+datestring+'.png'), dpi=600)
        plt.close()
      
      feature_importances_pandas2(self.forest_clf_poly, self.X_newdata_train_poly, feature_names, self.newdata_minusEP)
     
    get_polynomials(self.X_newdata_transform_train, self.y_train)      

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  newdata_minusEP, bbbb= make_output_folder(args.outdir)

  ###############################################################################

  polynomial_features = CreatePolynomialFeatures(metrix, newdata_minusEP, bbbb)

