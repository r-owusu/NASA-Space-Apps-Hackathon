import os, joblib
from pathlib import Path

DEFAULT_MODELS_DIR = 'saved_models'

def list_model_files(models_dir=DEFAULT_MODELS_DIR):
    """List all available model files from multiple possible locations"""
    model_files = []
    
    # Check local saved_models directory
    if os.path.exists(models_dir):
        local_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        model_files.extend([(f, models_dir) for f in local_models])
    
    # Check stage2 models directory (relative to current file)
    current_dir = Path(__file__).parent
    stage2_models_dir = current_dir.parent / "stage2_model_training" / "stage2_outputs" / "saved_models"
    
    if stage2_models_dir.exists():
        stage2_models = [f.name for f in stage2_models_dir.glob("*.pkl")]
        model_files.extend([(f, str(stage2_models_dir)) for f in stage2_models])
    
    # Return just the filenames for display, but keep track of paths
    return model_files

def load_model_by_name(name, models_dir=DEFAULT_MODELS_DIR):
    """Load model from the correct directory"""
    # First try local directory
    local_path = os.path.join(models_dir, name)
    if os.path.exists(local_path):
        obj = joblib.load(local_path)
        if isinstance(obj, dict): 
            return obj.get('model'), obj.get('scaler'), obj.get('metadata')
        return obj, None, None
    
    # Try stage2 directory
    current_dir = Path(__file__).parent
    stage2_path = current_dir.parent / "stage2_model_training" / "stage2_outputs" / "saved_models" / name
    
    if stage2_path.exists():
        obj = joblib.load(stage2_path)
        if isinstance(obj, dict): 
            return obj.get('model'), obj.get('scaler'), obj.get('metadata')
        return obj, None, None
    
    raise FileNotFoundError(f"Model '{name}' not found in any directory")

def get_model_info(name, models_dir=DEFAULT_MODELS_DIR):
    """Get information about a model without loading it"""
    try:
        model, scaler, metadata = load_model_by_name(name, models_dir)
        info = {
            'has_model': model is not None,
            'has_scaler': scaler is not None,
            'has_metadata': metadata is not None,
            'metadata': metadata if metadata else {}
        }
        return info
    except Exception as e:
        return {'error': str(e), 'has_model': False}
