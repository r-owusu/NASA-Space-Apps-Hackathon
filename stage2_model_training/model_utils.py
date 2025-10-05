import os, json, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,roc_curve

def evaluate_model(clf, X, y):
    preds = clf.predict(X)
    if hasattr(clf,'predict_proba'):
        probs = clf.predict_proba(X)[:,1]
    else:
        try:
            scores = clf.decision_function(X)
            probs = 1/(1+np.exp(-scores))
        except:
            probs = preds.astype(float)
    metrics = {'accuracy':float(accuracy_score(y,preds)),'precision':float(precision_score(y,preds,zero_division=0)),
               'recall':float(recall_score(y,preds,zero_division=0)),'f1':float(f1_score(y,preds,zero_division=0)),
               'roc_auc':float(roc_auc_score(y,probs)) if len(np.unique(y))>1 else 0.0}
    cm = confusion_matrix(y,preds)
    return metrics,preds,probs,cm

def plot_roc(y_true,y_score,out_path):
    fpr,tpr,_ = roc_curve(y_true,y_score)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--',color='gray'); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_confusion_matrix(cm,out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4)); plt.imshow(cm,cmap='Blues'); plt.colorbar(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_feature_importance(names,importances,out_path,top_k=30):
    import pandas as pd, matplotlib.pyplot as plt
    importances = pd.Series(importances,index=names).sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(6, max(4,0.2*len(importances)))); importances.plot(kind='barh'); plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def save_model_and_metadata(clf, scaler, model_path, scaler_path, metadata_path, metadata):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model':clf,'scaler':scaler,'metadata':metadata}, model_path)
    with open(metadata_path,'w') as fh: json.dump(metadata,fh,indent=2)
