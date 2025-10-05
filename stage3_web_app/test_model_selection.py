import streamlit as st
import os
import sys

st.title("🔬 Model Selection Test")

st.write("### Debug Information")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Python path: {sys.path}")

# Check saved_models directory
models_dir = "saved_models"
st.write(f"### Checking '{models_dir}' directory:")

if os.path.exists(models_dir):
    st.success(f"✅ Directory '{models_dir}' exists")
    
    files = os.listdir(models_dir)
    st.write(f"Files found: {files}")
    
    model_files = [f for f in files if f.endswith('.pkl')]
    st.write(f"Model files (.pkl): {model_files}")
    
    if model_files:
        st.write("### Model Selection Test")
        
        # Test the selectbox
        selected = st.selectbox(
            "Choose a model:",
            ["(Select a model)"] + model_files,
            key="test_selector"
        )
        
        st.write(f"Selected: {selected}")
        
        if selected and selected != "(Select a model)":
            st.success(f"✅ Model selected: {selected}")
            
            # Test loading
            if st.button("Test Load Model"):
                try:
                    from model_loader import load_model_by_name
                    model, scaler, metadata = load_model_by_name(selected)
                    st.success("✅ Model loaded successfully!")
                    st.write(f"Model type: {type(model)}")
                    st.write(f"Scaler type: {type(scaler)}")
                    st.write(f"Metadata: {metadata}")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
                    st.exception(e)
        else:
            st.info("No model selected yet")
    else:
        st.warning("⚠️ No .pkl files found in saved_models directory")
else:
    st.error(f"❌ Directory '{models_dir}' does not exist")
    st.write("Creating directory...")
    os.makedirs(models_dir, exist_ok=True)
    st.info("Directory created. Please add model files.")

# Session state debug
st.write("### Session State")
st.write(st.session_state)