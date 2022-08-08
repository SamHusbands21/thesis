#this code creates bootstrapped datasets of mortality.csv and mortality_large.csv for later model_building
# #packages
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
import os
np.random.seed(21)

def bootstrapper(df, file_location, bootstrap_count = 999, n = None):
    '''This function takes a dataset and produces bootstrap draws of it
       df: The dataframe to be bootstrapped (pandas dataframe)
       file_location: A string showing where to save the bootstrapped dataset
       bootstrap_count: the number a bootrap draws (integer; default 999)
       n: The number of draws of the orginal dataset with replacement (by default the length of original dataset)'''
    if n == None:
        n = len(df.index)
    for i in range(bootstrap_count):
        data = df.iloc[np.random.randint(n, size = n)].reset_index(drop = False)
        data.to_csv(file_location + '/mortality_{}.csv'.format(str(i+1)), index = False)
    return None

np.random.seed(21)
mf = pd.read_csv('./data/mortality.csv').set_index('id')
if os.path.exists('./data/bootstraps') == False:
    os.mkdir('./data/bootstraps')
bootstrapper(mf, file_location = './data/bootstraps', bootstrap_count = 999)

np.random.seed(21)
mf = pd.read_csv('./data/mortality_large.csv').set_index('SEQN')
if os.path.exists('./data/large_bootstraps') == False:
    os.mkdir('./data/large_bootstraps')
bootstrapper(mf, file_location = './data/large_bootstraps', bootstrap_count = 199)

np.random.seed(21)
mf = pd.read_csv('./data/mortality.csv').set_index('id')
if os.path.exists('./data/hp_bootstraps') == False:
    os.mkdir('./data/hp_bootstraps')
bootstrapper(mf, file_location = './data/hp_bootstraps', bootstrap_count = 199)