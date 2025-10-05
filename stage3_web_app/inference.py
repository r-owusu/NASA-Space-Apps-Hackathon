import numpy as np, pandas as pd
REQUIRED_BASE_COLS = ['dt','dtheta','strength_ratio']

def validate_input_df(df):
    missing=[c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing: raise ValueError(f'Missing columns: {missing}')
    return df

def derive_features(df):
    df = df.copy()
    df['dt']=pd.to_numeric(df['dt'],errors='coerce').fillna(0.0)
    df['dtheta']=pd.to_numeric(df['dtheta'],errors='coerce').fillna(9999.0)
    df['strength_ratio']=pd.to_numeric(df['strength_ratio'],errors='coerce').fillna(0.0)
    df['log_strength_ratio']=np.sign(df['strength_ratio'])*np.log1p(np.abs(df['strength_ratio']))
    return df

def prepare_features_for_model(df, scaler=None):
    # Start with base features
    base_features = ['dt','dtheta','strength_ratio','log_strength_ratio']
    X_df = df[base_features].astype(float).fillna(0.0)
    
    # Add pair features if m1 and m2 columns exist
    if 'm1' in df.columns and 'm2' in df.columns:
        df_copy = df.copy()
        df_copy['pair'] = df_copy['m1'].astype(str) + '_' + df_copy['m2'].astype(str)
        pair_dummies = pd.get_dummies(df_copy['pair'], prefix='pair', drop_first=True)
        X_df = pd.concat([X_df, pair_dummies], axis=1)
    
    # For inference without pair info, create dummy pair columns to match training
    # These are the exact feature names the model expects (from scaler.feature_names_in_)
    expected_pairs = ['pair_GW_Neutrino', 'pair_GW_Optical', 'pair_GW_Radio', 
                     'pair_Gamma_Neutrino', 'pair_Gamma_Optical', 'pair_Gamma_Radio',
                     'pair_Neutrino_Optical', 'pair_Neutrino_Radio', 'pair_Optical_Radio']
    
    # Ensure all expected pair columns exist (add missing ones as zeros)
    for pair in expected_pairs:
        if pair not in X_df.columns:
            X_df[pair] = 0
    
    # Reorder columns to match training order
    all_expected_features = ['dt', 'dtheta', 'strength_ratio', 'log_strength_ratio'] + expected_pairs
    X_df = X_df[all_expected_features]
    
    feature_names = X_df.columns.tolist()
    if scaler is not None:
        return scaler.transform(X_df.values), feature_names
    return ((X_df.values - X_df.mean().values)/(X_df.std().values+1e-9)), feature_names

def predict_df(df, model, scaler=None, threshold=0.5):
    df = validate_input_df(df)
    df = derive_features(df)
    X, _ = prepare_features_for_model(df, scaler)
    if hasattr(model,'predict_proba'):
        probs = model.predict_proba(X)[:,1]
    else:
        if hasattr(model,'decision_function'):
            probs = 1/(1+np.exp(-model.decision_function(X)))
        else:
            probs = model.predict(X).astype(float)
    df['pred_prob']=probs; df['pred_label']=(df['pred_prob']>=threshold).astype(int)
    return df
