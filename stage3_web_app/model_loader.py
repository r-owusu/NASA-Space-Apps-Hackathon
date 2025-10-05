import os, joblib
DEFAULT_MODELS_DIR='saved_models'

def list_model_files(models_dir=DEFAULT_MODELS_DIR):
    if not os.path.exists(models_dir): return []
    return sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')])

def load_model_by_name(name, models_dir=DEFAULT_MODELS_DIR):
    path = os.path.join(models_dir, name)
    if not os.path.exists(path): raise FileNotFoundError(path)
    obj = joblib.load(path)
    if isinstance(obj, dict): return obj.get('model'), obj.get('scaler'), obj.get('metadata')
    return obj, None, None
