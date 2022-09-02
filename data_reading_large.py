#this code ingests data from over 100 candidate data files (listed in datset_listing.md) and outputs mortality_large.csv
# #relevant packages
import pandas as pd
import numpy as np
import glob

#creating a list of all files from 2007-08 NHANES
all_filenames = glob.glob('./data/raw_data_files/*.{}'.format('XPT'))
all_filenames = [s.replace('\\', '/') for s in all_filenames]

#loading the mortality file (first ran through enclosed R script)
#and formatting it properly
lmf = pd.read_csv('./data/raw_data_files/lmf.csv')
sel_lmf = lmf.copy()[['seqn', 'eligstat', 'mortstat', 'permth_exm']]
sel_lmf.rename(columns = {'seqn': 'SEQN',
                          'eligstat': 'eligible',
                          'mortstat': 'deceased',
                          'permth_exm': 'months'},
               inplace = True)
sel_lmf.set_index('SEQN', inplace = True)
sel_lmf['eligible'] = sel_lmf['eligible'].apply(
    lambda x: "Eligible" if x == 1 else ("Ineligible" if x == 2 else np.NaN))
sel_lmf['deceased'] = np.where(sel_lmf['eligible']=='Eligible',
                             np.logical_and(sel_lmf['deceased'] == 1, sel_lmf['months'] <= 120),
                             np.NaN)
base_df = sel_lmf[['deceased']].dropna(axis = 0, how = 'any')

#how many candidate features (excluding dietary)
var_names = []
for filename in all_filenames:
    temp = pd.read_sas(filename).copy()
    temp_varnames = list(temp.columns)
    var_names += temp_varnames
print(len(set(var_names)))

#this code iterates through each file. If there is no sequence number,
#or there are multiple instances of the same sequence number,
#the data was not considered.
dataframes = []
for filename in all_filenames:
    temp = pd.read_sas(filename).copy()
    if 'SEQN' not in (list(temp.columns)):
        continue
    if len(temp['SEQN']) != len(set(temp['SEQN'])):
        continue
    temp[['SEQN']] = temp[['SEQN']].astype(int)
    temp.set_index('SEQN', inplace = True)
    dataframes.append(temp)

# this code iterates over every single remaining feature.
# if the feature has less than 10% missing data, (after accounitng for mortality missing data) it is selected.
filtered_dataframes = [base_df]
for df in dataframes:
    tester_df = base_df.merge(df, left_index=True, right_index=True, how='left')
    summarised_tester = tester_df.iloc[:, 1:].isna().sum().div(62.91)
    filtered_tester = summarised_tester[summarised_tester <= 10]
    if len(list(filtered_tester.index)) > 0:
        filtered_dataframes.append(df[list(filtered_tester.index)])

def df_column_uniquify(df):
    '''A simple function that iterates through column names and addresses duplicates, adding a counter to
       duplicate column names'''
    df_columns = df.columns
    new_columns = []
    potential_duplicates = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            potential_duplicates.append(newitem)
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

#merging the dataframes and making it complete case
output_df = pd.concat(filtered_dataframes, axis = 1).dropna(axis = 0, subset = ['deceased'])
output_df.dropna(axis = 0, inplace = True)
output_df = df_column_uniquify(output_df)
print(len(list(output_df.columns)))
print(output_df)
#There are 342 valid features, some of which are plainly categorical and require one hot encoding

def int_converter(df):
    '''Converts any columns that are integers into integers, not floats'''
    temp_df = df.copy()
    for col in list(df.columns):
        if temp_df[col].dtypes != 'float64':
            continue
        temp_df[col] = temp_df[col]%1
        if temp_df[col].sum() == 0:
            df[col] = df[col].astype('int64')
    return df

output_df = int_converter(output_df)
cat = []
cont = []
invest  = []
for col in output_df:
    print(col, len(output_df[col].unique()), output_df[col].dtypes)
    if len(output_df[col].unique()) <=20 and output_df[col].dtypes == 'int64':
        cat.append(col)
    elif len(output_df[col].unique()) >20 and output_df[col].dtypes == 'float64':
        cont.append(col)
    else:
        invest.append(col)
#simple means of seperating categorical data from continuous (some was decided manually)

invest_cont = ['URXUCR', 'URXCRS', 'PEASCTM1', 'BPXPLS', 'LBDMONO', 'LBDEONO', 'LBDBANO', 'LBXPLTSI', 'CBD150', 'CBD160',
               'URXUCR_1', 'RIDAGEYR', 'RIDAGEMN', 'RIDAGEEX', 'DMDHRAGE', 'LBXRBFSI', 'URXUCR_2', 'URXUCR_3', 'WHD010',
               'WHD020', 'WHD050', 'WHD140', 'WHQ150']
invest_cat = ['DBQ197', 'PHDSESN', 'HUQ050', 'LBDIHGLC', 'LBDTHGLC', 'URDUP8LC', 'URDNO3LC', 'URDSCNLC', 'SLQ030',
              'SLQ040', 'SLQ080', 'SLQ090', 'SLQ100', 'SLQ110', 'SLQ120', 'SLQ130','SLQ140', 'SLQ150', 'SLQ160',
              'SPXNQFVC', 'SPXNQFV1', 'SPXNQEFF', 'VIQ041', 'VIQ061']
invest_invalid = ['SPXBQFVC', 'SPXBQFV1', 'SPXBQEFF', 'SMDUPCA', 'SMD100BR']
#manually deciding if these 52 features were continuous, categorical or not valid
#(due to missing data) using NHANES documentation

for col in invest_cat:
    if output_df[col].dtypes == 'float64':
        output_df[col] = output_df[col].astype('int64')
#converting those columns to integers that are categorical and did not work previously

cont += invest_cont
cat += invest_cat
output_df.drop(labels = invest_invalid, axis = 1, inplace = True)

for col in list(output_df.columns):
    if len(output_df[col].unique())==1:
        output_df.drop(col, axis = 1, inplace = True)
print(len(output_df.columns))

#one hot encoding of categoricals
for col in cat:
    if col not in output_df.columns:
        cat.remove(col)
        continue
    output_df[col] = output_df[col].astype('object')
cat.remove('RIDSTATR')
cat.remove('VIQ130')
bin_vars = output_df[cat].copy()
bin_vars = pd.get_dummies(bin_vars, drop_first = True)
bin_vars.rename(columns = {'deceased_1': 'deceased'},
                inplace = True)

final_df = bin_vars.merge(output_df[cont], left_index = True, right_index = True, how = 'left')
final_df.to_csv('./data/mortality_large.csv')