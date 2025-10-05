import pandas as pd
import numpy as np
from model_loader import load_model_by_name
from inference import predict_df

print("=== Testing App Workflow ===")

# Create sample data exactly like the app
rng = np.random.RandomState(0)
n_samples = 10
messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
pairs = [(m1, m2) for i, m1 in enumerate(messengers) for m2 in messengers[i+1:]]
pair_choices = [pairs[i % len(pairs)] for i in range(n_samples)]
m1_list = [p[0] for p in pair_choices]
m2_list = [p[1] for p in pair_choices]

df = pd.DataFrame({
    'm1': m1_list,
    'm2': m2_list,
    'dt': rng.normal(0, 50, n_samples),
    'dtheta': np.abs(rng.normal(5, 3, n_samples)),
    'strength_ratio': np.abs(rng.normal(1, 0.8, n_samples)),
    'ra': rng.uniform(0, 360, n_samples),
    'dec': rng.uniform(-90, 90, n_samples)
})

print("✅ Sample data created")
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Test model loading
try:
    model, scaler, metadata = load_model_by_name('best_model.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)

# Test inference
try:
    threshold = 0.5
    df_pred = predict_df(df, model, scaler=scaler, threshold=threshold)
    print("✅ Inference successful!")
    
    # Test metrics calculation
    total_events = len(df_pred)
    high_confidence = len(df_pred[df_pred['pred_prob'] >= 0.8])
    positive_associations = len(df_pred[df_pred['pred_label'] == 1])
    
    print(f"📊 Total events: {total_events}")
    print(f"🔴 High confidence: {high_confidence}")
    print(f"✅ Positive associations: {positive_associations}")
    print(f"📈 Sample predictions: {df_pred['pred_prob'].head().tolist()}")
    
except Exception as e:
    print(f"❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()

print("=== Test Complete ===")