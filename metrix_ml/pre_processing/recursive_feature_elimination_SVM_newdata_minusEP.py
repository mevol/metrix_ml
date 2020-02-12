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
  names = ['newdata_minusEP', 'bbbb']
  result = []
  for name in names:
    name = os.path.join(outdir, 'recursive_feature_elimination', name)
    os.makedirs(name, exist_ok=True)
    result.append(name)
  return result

###############################################################################
#
#  class for ML using random forest with randomised search
#
###############################################################################

class RecursiveFeatureElimination(object):
  '''This class is the doing the actual work in the following steps:
     * define smaller data frames: database, man_add, transform
     * split the data into training and test set
     * setup and run a grid search for best paramaters to define a random forest
     * create a new random forest with best parameters
     * predict on this new random forest with test data and cross-validated training data
     * analyse the predisctions with graphs and stats
  '''
  def __init__(self, metrix, newdata_minusEP, bbbb):
    self.metrix=metrix
    self.newdata_minusEP=newdata_minusEP
    self.prepare_metrix_data()
    self.split_data()
    self.run_rfecv()

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
    Output: smaller dataframes; database
    '''
    print('*' *80)
    print('*    Preparing input dataframe metrix_database')
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

#    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
#                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
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
                      
    attr_newdata_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
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
    self.X_newdata_initial = metrix_newdata_initial

    metrix_newdata_transform = metrix_newdata_initial.copy()
    
    with open(os.path.join(self.newdata_minusEP, 'recursive_feature_elimination.txt'), 'a') as text_file:
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
    
    #self.X_newdata_transform.to_csv(os.path.join(self.newdata, 'transformed_dataframe.csv'))
    
    #np.isnan(self.X_newdata_transform)
    #print(np.where(np.isnan(self.X_newdata_transform)))
    #self.X_newdata_transform = np.nan_to_num(self.X_newdata_transform)
    self.X_newdata_transform = self.X_newdata_transform.fillna(0)
    
    with open(os.path.join(self.newdata_minusEP, 'recursive_feature_elimination.txt'), 'a') as text_file:
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

    with open(os.path.join(self.newdata_minusEP, 'recursive_feature_elimination.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_newdata_transform: X_newdata_transform_train, X_newdata_transform_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')


###############################################################################
    
    #standardise data
    X_newdata_transform_train_std = StandardScaler().fit_transform(self.X_newdata_transform_train)
    self.X_transform_newdata_transform_std = X_newdata_transform_train_std

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
      rfecv = RFECV(estimator=svc, step=1, min_features_to_select=5, cv=StratifiedKFold(3), scoring='accuracy')
      rfecv.fit(X_train, y)
      
      feature_names = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
       'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations', 'totalunique',
       'multiplicity', 'completeness', 'lowreslimit', 'highreslimit',
       'wilsonbfactor', 'anomalousslope', 'anomalousCC', 'anomalousmulti',
       'anomalouscompl', 'diffI', 'diffF', 'f', 'wavelength', 'sg_number', 'cell_a',
       'cell_b', 'cell_c', 'cell_alpha', 'cell_beta', 'cell_gamma', 'Vcell',
       'solvent_content', 'Matth_coeff', 'No_atom_chain', 'No_mol_ASU',
       'MW_chain', 'sites_ASU', 'MW_ASU',
       'MW_ASU/sites_ASU', 'IoverSigma/MW_ASU', 'MW_chain/No_atom_chain',
       'MW_ASU/sites_ASU/solvent_content', 'Vcell/Vm<Ma>', 'bragg',
       'volume_wilsonB_highres']
      
      feature_selection = rfecv.support_
      with open(os.path.join(self.newdata_minusEP, 'recursive_feature_elimination.txt'), 'a') as text_file:
        text_file.write('Feature mask : %s \n' % feature_selection)     
      #print(feature_selection)
      
      feature_ranking = rfecv.ranking_
      with open(os.path.join(self.newdata_minusEP, 'recursive_feature_elimination.txt'), 'a') as text_file:
        text_file.write('Feature ranking : %s \n' % feature_ranking)      
        text_file.write('Features sorted by their rank: \n')
        text_file.write(str(sorted(zip(map(lambda x: round(x, 4), feature_ranking), feature_names))))
     
      
      #print(feature_ranking)
      
      print("Features sorted by their rank:")
      print(sorted(zip(map(lambda x: round(x, 4), feature_ranking), feature_names)))
            
      print("Optimal number of features : %d" % rfecv.n_features_)
      
      with open(os.path.join(self.newdata_minusEP, 'recursive_feature_elimination.txt'), 'a') as text_file:
        text_file.write('Optimal number of features : %d \n' % rfecv.n_features_)

      # Plot number of features VS. cross-validation scores
      plt.figure()
      plt.xlabel("Number of features selected")
      plt.ylabel("Cross validation score (nb of correct classifications)")
      plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
      plt.savefig(os.path.join(self.newdata_minusEP, 'num_features_to_use_'+datestring+'.png'), dpi=600)     
      plt.close()
      
    run_rfecv_and_plot(self.X_transform_newdata_transform_std, self.y_train)
      

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)
  
  newdata_minusEP, bbbb= make_output_folder(args.outdir)

  ###############################################################################

  feature_decomposition = RecursiveFeatureElimination(metrix, newdata_minusEP, bbbb)
      


