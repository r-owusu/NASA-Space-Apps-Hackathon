#!/usr/bin/env python3
import os, json, time, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

from preprocess_data import load_pairwise_csvs, feature_engineering, split_and_scale
import model_utils as mu

def find_data_dir():
    candidates = ['../dataset_generation/multimessenger_data','../multimessenger_data','../stage1_dataset_generation/synthetic_multimessenger_data','synthetic_multimessenger_data']
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError('No dataset directory found among candidates: '+str(candidates))

def main(sample=0, seed=42, out_dir='stage2_outputs', fast=False, undersample=False):
    np.random.seed(seed)
    DATA_DIR = find_data_dir()
    print('Using DATA_DIR=',DATA_DIR)
    df = load_pairwise_csvs(DATA_DIR, sample_n=(sample if sample>0 else None), random_state=seed)
    print('Total rows:', len(df))
    X,y = feature_engineering(df)
    print('Feature matrix shape:', X.shape, 'Positives:', int(y.sum()), 'Negatives:', int((~y.astype(bool)).sum()))
    if undersample:
        df_comb = pd.concat([X, y.rename('label')], axis=1)
        pos = df_comb[df_comb.label==1]; neg = df_comb[df_comb.label==0]
        if len(pos)==0:
            print('No positive examples; aborting.'); return
        neg_sampled = neg.sample(n=len(pos), random_state=seed)
        df_bal = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=seed)
        y = df_bal.label; X = df_bal.drop(columns=['label'])
        print('After undersample shape:', X.shape)
    data = split_and_scale(X,y,random_state=seed)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    scaler = data['scaler']

    os.makedirs(out_dir, exist_ok=True); results_dir = os.path.join(out_dir,'results'); models_dir = os.path.join(out_dir,'saved_models')
    os.makedirs(results_dir, exist_ok=True); os.makedirs(models_dir, exist_ok=True)

    rf_grid = {'n_estimators':[100],'max_depth':[10]}
    lr_grid = {'C':[0.1,1.0]}
    models_summary = {}
    print('Training LogisticRegression...')
    lr = LogisticRegression(max_iter=400)
    gs_lr = GridSearchCV(lr, lr_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    try:
        gs_lr.fit(X_train, y_train)
    except Exception as e:
        print('LogisticRegression training failed:', e)
        gs_lr = None
    if gs_lr:
        best_lr = gs_lr.best_estimator_
        metrics_val,_,probs_val,cm = mu.evaluate_model(best_lr, X_val, y_val)
        metrics_test,_,probs_test,cm_test = mu.evaluate_model(best_lr, X_test, y_test)
        models_summary['LogisticRegression'] = {'val':metrics_val,'test':metrics_test,'best_params':gs_lr.best_params_}

    print('Training RandomForest...')
    rf = RandomForestClassifier(random_state=seed)
    gs_rf = GridSearchCV(rf, rf_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    try:
        gs_rf.fit(X_train, y_train)
    except Exception as e:
        print('RandomForest training failed:', e)
        gs_rf = None
    if gs_rf:
        best_rf = gs_rf.best_estimator_
        metrics_val,_,rf_probs_val,cm_rf = mu.evaluate_model(best_rf, X_val, y_val)
        metrics_test,_,rf_probs_test,cm_rf_test = mu.evaluate_model(best_rf, X_test, y_test)
        models_summary['RandomForest'] = {'val':metrics_val,'test':metrics_test,'best_params':gs_rf.best_params_}

    best_name=None; best_auc=-1.0; best_model=None
    for name,info in models_summary.items():
        auc = info['val'].get('roc_auc',0.0)
        if auc>best_auc:
            best_auc=auc; best_name=name; best_model = best_lr if name=='LogisticRegression' else best_rf
    if best_model is not None:
        metadata = {'best_model':best_name,'best_auc':best_auc,'date':int(time.time())}
        mu.save_model_and_metadata(best_model, scaler, os.path.join(models_dir,'best_model.pkl'), os.path.join(models_dir,'scaler.pkl'), os.path.join(models_dir,'metadata.json'), metadata)
        with open(os.path.join(results_dir,'model_metrics.json'),'w') as fh: json.dump(models_summary, fh, indent=2)
        print('Saved best model and metrics.')
    else:
        print('No model trained successfully.')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='stage2_outputs')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--undersample', action='store_true')
    args = parser.parse_args()
    main(sample=args.sample, seed=args.seed, out_dir=args.out_dir, fast=args.fast, undersample=args.undersample)
