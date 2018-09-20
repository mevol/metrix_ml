# set up environment
# define command line parameters
# define location of input data
# create output directories
# start the class FeatureCorrelations


import pandas as pd
import os
import numpy as np
import csv
from pandas.plotting import scatter_matrix
from datetime import datetime
from scipy.stats import pearsonr, betai
import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################
#
#  load the data from CSV file and creating output directory
#
###############################################################################

def load_metrix_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path)

def make_output_folder(outdir):
  names = ['database', 'man_add', 'transform', 'prot_screen_trans']
  result = []
  for name in names:
    name = os.path.join(outdir, 'decisiontree_randomsearch', name)
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
#    def __init__(self, X_train):
#        if X_train.isnull().any().any() == True:
#            X_train = X_train.dropna(axis=1)
#        pass


 
  def __init__(self, metrix, database, man_add, transform, prot_screen_trans):
    self.metrix=metrix
    self.database=database
    self.man_add=man_add
    self.transform=transform
    self.prot_screen_trans=prot_screen_trans
    self.prepare_metrix_data()
    self.split_data()
   
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

    #look at the data that is coming from processing
    attr_database = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                      'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                      'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                      'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF']
    metrix_database = self.metrix[attr_database]
    
    with open(os.path.join(self.database, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_database with following attributes %s \n' %(attr_database))

    #database plus manually added data
    attr_man_add = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                    'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                    'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                    'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF',
                    'wavelength', 'Vcell', 'Matth_coeff', 'No_atom_chain', 'solvent_content',
                    'No_mol_ASU', 'MW_chain', 'sites_ASU']
    metrix_man_add = self.metrix[attr_man_add]

    with open(os.path.join(self.man_add, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_man_add with following attributes %s \n' %(attr_man_add))

    #after column transformation expected feature list
    attr_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
                      'diffF', 'wavelength', 'wavelength**3', 'wavelength**3/Vcell',
                      'Vcell', 'solvent_content', 'Vcell/Vm<Ma>', 'Matth_coeff',
                      'MW_ASU/sites_ASU/solvent_content', 'MW_chain', 'No_atom_chain',
                      'No_mol_ASU', 'MW_ASU', 'sites_ASU', 'MW_ASU/sites_ASU',
                      'MW_chain/No_atom_chain', 'wilson', 'bragg',
                      'volume_wilsonB_highres']

    attr_prot_screen_trans = ['highreslimit', 'wavelength', 'Vcell', 'wavelength**3',
                         'wavelength**3/Vcell', 'solvent_content', 'Vcell/Vm<Ma>',
                         'Matth_coeff', 'MW_ASU/sites_ASU/solvent_content',
                         'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU',
                         'sites_ASU', 'MW_ASU/sites_ASU', 'MW_chain/No_atom_chain']

    metrix_transform = metrix_man_add.copy()
    metrix_prot_screen_trans = metrix_man_add[['highreslimit', 'wavelength', 'Vcell',
    'Matth_coeff', 'No_atom_chain', 'solvent_content', 'No_mol_ASU', 'MW_chain', 'sites_ASU']].copy()

    with open(os.path.join(self.transform, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_transform with following attributes %s \n' %(attr_transform))

    with open(os.path.join(self.prot_screen_trans, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_transform with following attributes %s \n' %(attr_prot_screen_trans))

    #column transformation
    #MW_ASU
    metrix_transform['MW_ASU'] = metrix_transform['MW_chain'] * metrix_transform['No_mol_ASU']
    metrix_prot_screen_trans['MW_ASU'] = metrix_prot_screen_trans['MW_chain'] * metrix_prot_screen_trans['No_mol_ASU']

    #MW_ASU/sites_ASU
    metrix_transform['MW_ASU/sites_ASU'] = metrix_transform['MW_ASU'] / metrix_transform['sites_ASU']
    metrix_prot_screen_trans['MW_ASU/sites_ASU'] = metrix_prot_screen_trans['MW_ASU'] / metrix_prot_screen_trans['sites_ASU']

    #MW_chain/No_atom_chain
    metrix_transform['MW_chain/No_atom_chain'] = metrix_transform['MW_chain'] / metrix_transform['No_atom_chain']
    metrix_prot_screen_trans['MW_chain/No_atom_chain'] = metrix_prot_screen_trans['MW_chain'] / metrix_prot_screen_trans['No_atom_chain']

    #MW_ASU/sites_ASU/solvent_content
    metrix_transform['MW_ASU/sites_ASU/solvent_content'] = metrix_transform['MW_ASU/sites_ASU'] / metrix_transform['solvent_content']
    metrix_prot_screen_trans['MW_ASU/sites_ASU/solvent_content'] = metrix_prot_screen_trans['MW_ASU/sites_ASU'] / metrix_prot_screen_trans['solvent_content']

    #wavelength**3
    metrix_transform['wavelength**3'] = metrix_transform['wavelength'] ** 3
    metrix_prot_screen_trans['wavelength**3'] = metrix_prot_screen_trans['wavelength'] ** 3

    #wavelenght**3/Vcell
    metrix_transform['wavelength**3/Vcell'] = metrix_transform['wavelength**3'] / metrix_transform['Vcell']
    metrix_prot_screen_trans['wavelength**3/Vcell'] = metrix_prot_screen_trans['wavelength**3'] / metrix_prot_screen_trans['Vcell']

    #Vcell/Vm<Ma>
    metrix_transform['Vcell/Vm<Ma>'] = metrix_transform['Vcell'] / (metrix_transform['Matth_coeff'] * metrix_transform['MW_chain/No_atom_chain'])
    metrix_prot_screen_trans['Vcell/Vm<Ma>'] = metrix_prot_screen_trans['Vcell'] / (metrix_prot_screen_trans['Matth_coeff'] * metrix_prot_screen_trans['MW_chain/No_atom_chain'])

    #wilson
    metrix_transform['wilson'] = -2 * metrix_transform['wilsonbfactor']

    #bragg
    metrix_transform['bragg'] = (1 / metrix_transform['highreslimit'])**2

    #use np.exp to work with series object
    metrix_transform['volume_wilsonB_highres'] = metrix_transform['Vcell/Vm<Ma>'] * np.exp(metrix_transform['wilson'] * metrix_transform['bragg'])

    self.X_database = metrix_database
    self.X_man_add = metrix_man_add
    self.X_transform = metrix_transform
    self.X_prot_screen_trans = metrix_prot_screen_trans

    with open(os.path.join(self.database, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_database \n')
      text_file.write(str(self.X_database.columns)+'\n')
    with open(os.path.join(self.man_add, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_man_add \n')
      text_file.write(str(self.X_man_add.columns)+'\n')     
    with open(os.path.join(self.transform, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_transform \n')
      text_file.write(str(self.X_transform.columns)+'\n')    
    with open(os.path.join(self.prot_screen_trans, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_prot_screen_trans \n')
      text_file.write(str(self.X_prot_screen_trans.columns)+'\n')

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
#    X_database_train, X_database_test, y_train, y_test = train_test_split(self.X_database, y, test_size=0.2, random_state=42)
#    X_man_add_train, X_man_add_test, y_train, y_test = train_test_split(self.X_man_add, y, test_size=0.2, random_state=42)
#    X_transform_train, X_transform_test, y_train, y_test = train_test_split(self.X_transform, y, test_size=0.2, random_state=42)
#    X_prot_screen_trans_train, X_prot_screen_trans_test, y_train, y_test = train_test_split(self.X_prot_screen_trans, y, test_size=0.2, random_state=42)

#stratified split of samples
    X_database_train, X_database_test, y_train, y_test = train_test_split(self.X_database, y, test_size=0.2, random_state=42, stratify=y)
    X_man_add_train, X_man_add_test, y_train, y_test = train_test_split(self.X_man_add, y, test_size=0.2, random_state=42, stratify=y)
    X_transform_train, X_transform_test, y_train, y_test = train_test_split(self.X_transform, y, test_size=0.2, random_state=42, stratify=y)
    X_prot_screen_trans_train, X_prot_screen_trans_test, y_train, y_test = train_test_split(self.X_prot_screen_trans, y, test_size=0.2, random_state=42, stratify=y)
    
    assert self.X_database.columns.all() == X_database_train.columns.all()
    assert self.X_man_add.columns.all() == X_man_add_train.columns.all()
    assert self.X_transform.columns.all() == X_transform_train.columns.all()
    assert self.X_prot_screen_trans.columns.all() == X_prot_screen_trans_train.columns.all()
    
    self.X_database_train = X_database_train
    self.X_man_add_train = X_man_add_train
    self.X_transform_train = X_transform_train
    self.X_prot_screen_trans_train = X_prot_screen_trans_train
    self.X_database_test = X_database_test
    self.X_man_add_test = X_man_add_test
    self.X_transform_test = X_transform_test
    self.X_prot_screen_trans_test = X_prot_screen_trans_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.database, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_database: X_database_train, X_database_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')
      
    with open(os.path.join(self.man_add, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_man_add: X_man_add_train, X_man_add_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')
      
    with open(os.path.join(self.transform, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_transform: X_transform_train, X_transform_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')

    with open(os.path.join(self.prot_screen_trans, 'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_transform: X_prot_screen_trans_train, X_prot_screen_trans_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')

    ###############################################################################
    #
    #  creating training and test set for each of the 3 dataframes
    #
    ###############################################################################

  def calculate_pearson_cc(X_train, name, directory):
    '''This functions calculates a simple statistics of
    Pearson correlation coefficient'''
    attr = list(X_train)
    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    for a in attr:
      corr_X_train = X_train.corr()        
      with open(os.path.join(directory, 'linear_PearsonCC_values_'+name+datestring+'.txt'), 'a') as text_file:
        corr_X_train[a].sort_values(ascending=False).to_csv(text_file)
      text_file.close()
        
  calculate_pearson_cc(self.X_database_train, 'database', self.database)
    
  def plot_scatter_matrix(X_train, name, directory):
    '''A function to create a scatter matrix of the data'''
    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    columns = [X_train.columns]
    for c in columns:
      if X_train.isnull().any().any() == True:
        X_train = X_train.dropna(axis=1)
        attr = list(X_train)    
        scatter_matrix(X_train[attr], figsize=(25,20))
        plt.savefig(os.path.join(directory, 'linear_PearsonCC_scattermatrix'+name+datestring+'.png'))
        plt.close()
  
  plot_scatter_matrix(self.X_database_train, 'database', self.database)

  def correlation_pvalue(X_train, name, directory):               
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
    return r, p
    
    r,p = corrcoef_loop(metrix_num)

    def write_corr(r, p, X_train, name, directory):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      with open(os.path.join(directory,
                           'linear_corr_pvalues_'+name+datestring+'.csv'), 'a') as csv_file:
        attr = list(X_train)
        rows = len(attr)
        for i in range(rows):
          for j in range(i+1, rows):
            csv_file.write('%s, %s, %f, %f\n' %(attr[i], attr[j], r[i,j], p[i,j]))
      csv_file.close()
    
    write_corr(r,p, X_train, name, directory)
  
  correlation_pvalue(self.X_database_train, 'database', self.database)
  
  def feature_conf_mat(X_train, name, directory):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = X_train.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(255, 10, as_cmap=True), square=True, ax=ax)
    plt.savefig(os.path.join(directory, 'feature_confusion_matrix_'+name+datestring+'.png'))

    plt.show()
    
  feature_conf_mat(self.X_database_train, 'database', self.database)
#  feature_conf_mat(self.X_man_add_train, 'man_add', self.man_add)
#  feature_conf_mat(self.X_transform_train, 'transform', self.transform)
#  feature_conf_mat(self.X_prot_screen_trans_train, 'prot_screen_trans', self.prot_screen_trans)
    
    
    
    
    
#confution matrix of features
#X_transform_train_ordered = X_transform_train[['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
#                          'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
#                          'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
#                          'highreslimit', 'wilsonbfactor', 'anomalousslope',
#                          'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
#                          'diffF', 'wavelength', 'wavelength**3', 'wavelength**3/Vcell',
#                          'Vcell', 'solvent_content', 'Vcell/Vm<Ma>', 'Matth_coeff',
#                          'MW_ASU/sites_ASU/solvent_content', 'MW_chain', 'No_atom_chain',
#                          'No_mol_ASU', 'MW_ASU', 'sites_ASU', 'MW_ASU/sites_ASU',
#                          'MW_chain/No_atom_chain', 'wilson', 'bragg', 'volume_wilsonB_highres']]                              

#X_transform_train_ordered.to_csv(os.path.join(METRIX_PATH, 'X_transform_train_ordered_forR.csv'), sep=',')



#f, ax = plt.subplots(figsize=(10, 8))
#corr = X_transform_train_ordered.corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(255, 10, as_cmap=True),
#            square=True, ax=ax)
#plt.show()
    
#print(X_transform_train.columns)
#print(X_transform_train_ordered.columns)

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  database, man_add, transform, prot_screen_trans= make_output_folder(args.outdir)

  ###############################################################################

  feature_correlations = FeatureCorrelations(metrix, database, man_add, transform, prot_screen_trans)
   
   
   
   
