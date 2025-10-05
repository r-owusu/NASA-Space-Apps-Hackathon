import streamlit as st
import os

st.title("🚀 Simple Model Selection Test")

st.write("## Step 1: Check Models Directory")

# List available models
models_dir = "saved_models"
if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    model_files = [f for f in files if f.endswith('.pkl')]
    
    st.success(f"Found {len(model_files)} model files:")
    for i, model in enumerate(model_files, 1):
        st.write(f"{i}. {model}")
    
    st.write("## Step 2: Test Model Selection")
    
    if model_files:
        # Simple selectbox test
        choice = st.selectbox(
            "Pick a model:",
            options=model_files,
            key="simple_select"
        )
        
        st.write(f"You selected: **{choice}**")
        
        if choice:
            st.success(f"✅ Successfully selected: {choice}")
            
            if st.button("✨ Confirm Selection"):
                st.balloons()
                st.success(f"🎉 Model {choice} is ready!")
        
    else:
        st.error("No model files found!")
        
else:
    st.error("saved_models directory not found!")

st.write("## Step 3: Debug Info")
st.write(f"Current directory: {os.getcwd()}")
st.write("Available files:")
for item in os.listdir("."):
    st.write(f"- {item}")