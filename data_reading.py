#packages
import pandas as pd
import numpy as np
import os

if os.path.exists('./data/raw_data_files') == False:
    print('This file structure is improperly setup')
#reading the various constituent pieces of the 2007-2008 NHANES, available on the website link described
demo = pd.read_sas('./data/raw_data_files/DEMO_E.XPT')
bm = pd.read_sas('./data/raw_data_files/BMX_E.XPT')
smk = pd.read_sas('./data/raw_data_files/SMQ_E.XPT')
alc = pd.read_sas('./data/raw_data_files/ALQ_E.XPT')
bp = pd.read_sas('./data/raw_data_files/BPX_E.XPT')
chol = pd.read_sas('./data/raw_data_files/TCHOL_E.XPT')
diab = pd.read_sas('./data/raw_data_files/DIQ_E.XPT')

#creates sel_demo, which takes the variables we are interested in
#from the demographic data file and recodes it in a more desireable format
sel_demo = demo.copy()[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
sel_demo.rename(columns = {'SEQN': 'id',
                           'RIDAGEYR': 'age',
                           'RIAGENDR': 'sex'},
                           inplace = True)
sel_demo[['id', 'age', 'sex']] = sel_demo[['id', 'age', 'sex']].astype(int)
sel_demo.set_index('id', inplace = True)
sel_demo['sex'] = sel_demo['sex'].apply(lambda x: "Male" if x == 1 else "Female")

#creates sel_bm, which takes the variable we are interested in
#from the body measurements data file and recodes it in a more desireable format
sel_bm = bm.copy()[['SEQN', 'BMXBMI']]
sel_bm.rename(columns = {'SEQN': 'id',
                         'BMXBMI': 'bmi'},
              inplace = True)
sel_bm[['id']] = sel_bm[['id']].astype(int)
sel_bm.set_index('id', inplace = True)

#creates sel_smk, which takes the variable we are interested in
#from the smoking - cigarette use data file and recodes it in a more desireable format
#smoker is a yes if more than 100 lifetime cigarettes
sel_smk = smk.copy()[['SEQN', 'SMQ020']]
sel_smk.rename(columns = {'SEQN': 'id',
                          'SMQ020': 'smoke'},
               inplace = True)
sel_smk[['id']] = sel_smk[['id']].astype(int)
sel_smk.set_index('id', inplace = True)
sel_smk['smoke'] = sel_smk['smoke'].apply(lambda x: "Yes" if x == 1.0 else ("No" if x == 2.0 else np.NaN))

#creates sel_alc, which takes the variable we are interested in
#from the alcohol use data file and recodes it in a more desireable format
#alcohol is a yes if consumed more than 12 alcoholic drinks per year.
sel_alc = alc.copy()[['SEQN', 'ALQ101']]
sel_alc.rename(columns = {'SEQN': 'id',
                          'ALQ101': 'alcohol'},
               inplace = True)
sel_alc[['id']] = sel_alc[['id']].astype(int)
sel_alc.set_index('id', inplace = True)
sel_alc['alcohol'] = sel_alc['alcohol'].apply(lambda x: "Yes" if x == 1.0 else ("No" if x == 2.0 else np.NaN))

#creates sel_bp, which takes the average of both systolic and diastolic blood pressures.
#measured in mmHg
sel_bp = bp.copy()[['SEQN', 'BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4', 'BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
sel_bp['systolic'] = sel_bp[['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']].mean(axis = 1, skipna = True)
sel_bp['diastolic'] = sel_bp[['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']].mean(axis = 1, skipna = True)
sel_bp = sel_bp[['SEQN', 'systolic', 'diastolic']]
sel_bp.rename(columns = {'SEQN': 'id'},
               inplace = True)
sel_bp[['id']] = sel_bp[['id']].astype(int)
sel_bp.set_index('id', inplace = True)

#creates sel_chol, which takes the variable we are interested in
#from the total cholesterol data file and recodes it in a more desirable format
#cholesterol in mmol/L
sel_chol = chol.copy()[['SEQN', 'LBDTCSI']]
sel_chol.rename(columns = {'SEQN': 'id',
                           'LBDTCSI': 'cholesterol'},
               inplace = True)
sel_chol[['id']] = sel_chol[['id']].astype(int)
sel_chol.set_index('id', inplace = True)

#creates sel_diab, which takes the variable we are interested in
#from the diabetes data file and recodes it in a more desirable format
#diabetes a yes if patient told they had diabetes (borderline treated as missing)
sel_diab = diab.copy()[['SEQN', 'DIQ010']]
sel_diab.rename(columns = {'SEQN': 'id',
                           'DIQ010': 'diabetes'},
               inplace = True)
sel_diab[['id']] = sel_diab[['id']].astype(int)
sel_diab.set_index('id', inplace = True)
sel_diab['diabetes'] = sel_diab['diabetes'].apply(lambda x: "Yes" if x == 1.0 else ("No" if x == 2.0 else np.NaN))

#loading the mortality file (first ran through enclosed R script)
#and formatting it properly
lmf = pd.read_csv('./data/raw_data_files/lmf.csv')
sel_lmf = lmf.copy()[['seqn', 'eligstat', 'mortstat', 'permth_exm']]
sel_lmf.rename(columns = {'seqn': 'id',
                          'eligstat': 'eligible',
                          'mortstat': 'deceased',
                          'permth_exm': 'months'},
               inplace = True)
sel_lmf.set_index('id', inplace = True)
sel_lmf['eligible'] = sel_lmf['eligible'].apply(
    lambda x: "Eligible" if x == 1 else ("Ineligible" if x == 2 else np.NaN))
sel_lmf['mort10'] = np.where(sel_lmf['eligible']=='Eligible',
                             np.logical_and(sel_lmf['deceased'] == 1, sel_lmf['months'] <= 120),
                             np.NaN)
sel_lmf['mort10'] = sel_lmf['mort10'].apply(lambda x: "Yes" if x == 1.0 else ("No" if x == 0 else np.NaN))

sel_lmf = sel_lmf[['eligible','mort10']]

#merges all the data together
df = sel_demo.copy()
df = df.join(sel_bm, how = 'left')
df = df.join(sel_smk, how = 'left')
df = df.join(sel_alc, how = 'left')
df = df.join(sel_bp, how = 'left')
df = df.join(sel_chol, how = 'left')
df = df.join(sel_diab, how = 'left')
df = df.join(sel_lmf, how = 'left')

#displaying summary
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df.describe())
print(df.sex.value_counts())
print(df.smoke.value_counts())
print(df.alcohol.value_counts())
print(df.diabetes.value_counts())
print(df.mort10.value_counts())

comp_df = df.dropna(axis = 0, how = 'any') #making it complete case

#displaying summary
print(comp_df.describe())
print(comp_df.sex.value_counts())
print(comp_df.smoke.value_counts())
print(comp_df.alcohol.value_counts())
print(comp_df.diabetes.value_counts())
print(comp_df.mort10.value_counts())

#one hot encoding
bin_vars = comp_df[['sex', 'smoke', 'alcohol', 'diabetes', 'eligible', 'mort10']].copy()
bin_vars = pd.get_dummies(bin_vars, drop_first = True)
bin_vars.rename(columns = {'sex_Male': 'male',
                           'smoke_Yes': 'smoker',
                           'alcohol_Yes': 'alcohol',
                           'diabetes_Yes': 'diabetes',
                           'mort10_Yes': 'deceased'},
                inplace = True)

comp_df = df[['age', 'bmi', 'systolic', 'diastolic', 'cholesterol']].copy().join(bin_vars, how = 'left')
comp_df.dropna(axis = 0, how = 'any', inplace = True)

comp_df.to_csv('./data/mortality.csv')