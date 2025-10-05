import os, glob
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DEFAULT_FEATURE_COLUMNS = ['dt','dtheta','strength_ratio']

def load_pairwise_csvs(data_dir, sample_n=None, random_state=42, verbose=True):
    pattern = os.path.join(data_dir, 'pairs_*.csv')
    files = sorted(glob.glob(pattern))
    if len(files)==0:
        raise FileNotFoundError(f'No pairwise files found with pattern: {pattern}')
    dfs=[]
    for f in files:
        if verbose: print('Loading',f)
        df = pd.read_csv(f)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    if sample_n and len(df_all)>sample_n:
        df_all = df_all.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    return df_all

def feature_engineering(df):
    df=df.copy()
    for c in ['dt','dtheta','strength_ratio','label']:
        if c not in df.columns: raise ValueError(f'Missing required column: {c}')
    df['log_strength_ratio'] = np.sign(df['strength_ratio'])*np.log1p(np.abs(df['strength_ratio']))
    if 'm1' in df.columns and 'm2' in df.columns:
        df['pair'] = df['m1'].astype(str)+'_'+df['m2'].astype(str)
    X = df[['dt','dtheta','strength_ratio','log_strength_ratio']].copy()
    if 'pair' in df.columns:
        X = pd.concat([X, pd.get_dummies(df['pair'], prefix='pair', drop_first=True)], axis=1)
    y = df['label'].astype(int).copy()
    return X, y

def split_and_scale(X,y,test_size=0.15,val_size=0.15,random_state=42):
    if len(np.unique(y))>1:
        X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
        val_relative = val_size/(1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp,y_temp,test_size=val_relative,random_state=random_state,stratify=y_temp)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=val_size/(1-test_size),random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return {'X_train':pd.DataFrame(X_train_s,columns=X_train.columns),'X_val':pd.DataFrame(X_val_s,columns=X_val.columns),'X_test':pd.DataFrame(X_test_s,columns=X_test.columns),'y_train':y_train,'y_val':y_val,'y_test':y_test,'scaler':scaler}
