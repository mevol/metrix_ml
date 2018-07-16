import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class ColumnTransformation():
    '''A class to run various column transformation steps to prepare the
    input until I am able to implement this step in the database code'''
    def __init__(self):
        pass

    def transform(self, df):
        #MW_ASU
        df['MW_ASU'] = df['MW_chain'] * df['No_mol_ASU']

        #MW_chain/No_atom_chain
        df['MW_chain/No_atom_chain'] = df['MW_chain'] / df['No_atom_chain']

        #wavelength**3
        df['wavelength**3'] = df['wavelength'] ** 3

        #wilson
        df['wilson'] = -2 * df['wilsonbfactor']

        #bragg
        df['bragg'] = (1 / df['highreslimit'])**2

        #MW_ASU/sites_ASU
        df['MW_ASU/sites_ASU'] = df['MW_ASU'] / df['sites_ASU']
        
        #wavelenght**3/Vcell
        df['wavelength**3/Vcell'] = df['wavelength**3'] / df['Vcell']
        
        #Vcell/Vm<Ma>
        df['Vcell/Vm<Ma>'] = df['Vcell'] / (df['Matth_coeff'] * df['MW_chain/No_atom_chain'])

        #MW_ASU/sites_ASU/solvent_content
        df['MW_ASU/sites_ASU/solvent_content'] = df['MW_ASU/sites_ASU'] / df['solvent_content']

        #use np.exp to work with series object
        df['volume_wilsonB_highres'] = df['Vcell/Vm<Ma>'] * np.exp(df['wilson'] * df['bragg'])

        return df.round(decimals=4).to_csv(os.path.join(METRIX_PATH, "data_transform.csv"))

#c = ColumnTransformation()
#c.transform(metrix)