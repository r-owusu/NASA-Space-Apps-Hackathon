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

def classify_event_confidence(prob, threshold=0.5):
    """Enhanced confidence classification with detailed reasoning"""
    if prob >= 0.9:
        return "Very High", "Strong evidence for same astronomical source"
    elif prob >= 0.75:
        return "High", "Likely from the same astronomical event"
    elif prob >= threshold:
        return "Moderate", "Probably associated with the same source"
    elif prob >= 0.25:
        return "Low", "Likely from different astronomical sources"
    else:
        return "Very Low", "Strong evidence for different sources"

def generate_physical_reasoning(row):
    """Generate physical reasoning for the classification"""
    dt = row.get('dt', 0)
    dtheta = row.get('dtheta', 0)
    strength_ratio = row.get('strength_ratio', 1)
    prob = row.get('pred_prob', 0.5)
    
    reasons = []
    
    # Time difference analysis
    if dt < 100:
        reasons.append("📅 Very short time difference suggests temporal coincidence")
    elif dt < 1000:
        reasons.append("⏱️ Moderate time separation allows for causal connection")
    else:
        reasons.append("⏰ Large time gap reduces likelihood of association")
    
    # Angular separation analysis
    if dtheta < 0.1:
        reasons.append("🎯 Excellent spatial coincidence (< 0.1°)")
    elif dtheta < 1.0:
        reasons.append("📍 Good spatial alignment within error bounds")
    elif dtheta < 5.0:
        reasons.append("🌐 Moderate spatial separation, possible association")
    else:
        reasons.append("📐 Large angular separation suggests different sources")
    
    # Strength ratio analysis
    if 0.5 <= strength_ratio <= 2.0:
        reasons.append("⚖️ Similar signal strengths support common origin")
    elif strength_ratio > 5.0:
        reasons.append("📊 Very different signal strengths may indicate different sources")
    else:
        reasons.append("📈 Moderate strength difference, inconclusive evidence")
    
    # Event type specific reasoning
    if 'm1' in row and 'm2' in row:
        event_types = f"{row['m1']}-{row['m2']}"
        if 'Gamma' in event_types and 'Neutrino' in event_types:
            reasons.append("🌟 Gamma-Neutrino pairs often associated with high-energy astrophysical processes")
        elif 'GW' in event_types:
            reasons.append("🌊 Gravitational wave events may have electromagnetic counterparts")
    
    return " • ".join(reasons)

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
    
    # Enhanced predictions with confidence and reasoning
    df['pred_prob'] = probs
    df['pred_label'] = (df['pred_prob'] >= threshold).astype(int)
    
    # Add enhanced classification
    confidence_data = [classify_event_confidence(p, threshold) for p in probs]
    df['confidence_level'] = [c[0] for c in confidence_data]
    df['confidence_description'] = [c[1] for c in confidence_data]
    
    # Add classification with emojis
    df['event_classification'] = df['pred_prob'].apply(
        lambda x: '✅ Same Astronomical Event' if x >= threshold else '❌ Different Events'
    )
    
    # Add detailed reasoning for each prediction
    df['physical_reasoning'] = df.apply(generate_physical_reasoning, axis=1)
    
    # Add risk assessment
    df['risk_assessment'] = df['pred_prob'].apply(
        lambda x: '🔴 High Risk of Misclassification' if 0.4 <= x <= 0.6 
        else '🟡 Moderate Confidence' if 0.25 <= x <= 0.75 
        else '🟢 High Confidence'
    )
    
    return df
