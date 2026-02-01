# ==========================================
# STREAMLIT APP - ENSEMBLE MRI CLASSIFICATION
# Using MobileNetV2, DenseNet121, and ResNet50
# ==========================================

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import traceback
import os

# ----------------------------
# CONSTANTS
# ----------------------------
IMG_SIZE = (224, 224)
MODEL_NAMES = ["MobileNetV2", "DenseNet121", "ResNet50"]

st.set_page_config(
    page_title="MRI Classification - Ensemble",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MRI Classification - Ensemble Model Prediction")
st.write("Upload an image to check if it is an MRI scan using 3 different deep learning models.")
st.write(f"TensorFlow version: {tf.__version__}")

# ----------------------------
# SIDEBAR - MODEL INFORMATION
# ----------------------------
st.sidebar.header("🔧 Model Information")

# ----------------------------
# LOAD ALL THREE MODELS
# ----------------------------
@st.cache_resource
def load_all_models():
    """Load all three trained models"""
    models = {}
    model_paths = {
        "MobileNetV2": "MobileNet_Binary_MRI.keras",
        "DenseNet121": "DenseNet121_Binary_MRI.keras",
        "ResNet50": "ResNet50_Binary_MRI.keras"
    }
    
    st.sidebar.write("**Loading Models:**")
    
    for model_name, model_path in model_paths.items():
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                st.sidebar.error(f"❌ {model_name}: File not found")
                st.error(f"🚨 Model file not found: {model_path}")
                continue
            
            # Get file size
            file_size = os.path.getsize(model_path) / 1024 / 1024
            st.sidebar.write(f"📂 {model_name}: {file_size:.2f} MB")
            
            # Load model
            model = tf.keras.models.load_model(model_path, compile=False)
            models[model_name] = model
            st.sidebar.success(f"✓ {model_name} loaded!")
            
        except Exception as e:
            st.sidebar.error(f"❌ {model_name} failed!")
            st.sidebar.code(str(e))
            st.error(f"🚨 Failed to load {model_name}: {str(e)}")
    
    return models

# Load all models
models_loaded = False
models = {}

try:
    st.sidebar.write("="*50)
    models = load_all_models()
    
    if len(models) == 0:
        st.error("🚨 **NO MODELS LOADED**")
        st.warning("⚠️ Please ensure all model files are in the correct directory.")
        st.stop()
    
    models_loaded = True
    st.sidebar.write("="*50)
    st.success(f"✅ Successfully loaded {len(models)}/{len(MODEL_NAMES)} models!")
    
    # Display loaded models
    st.sidebar.write("\n**Loaded Models:**")
    for model_name in models.keys():
        st.sidebar.write(f"  ✓ {model_name}")
    
    if len(models) < len(MODEL_NAMES):
        missing_models = set(MODEL_NAMES) - set(models.keys())
        st.sidebar.write("\n**Missing Models:**")
        for model_name in missing_models:
            st.sidebar.write(f"  ✗ {model_name}")
        st.warning(f"⚠️ Only {len(models)} out of {len(MODEL_NAMES)} models loaded. Results may be incomplete.")
    
except Exception as e:
    st.error("🚨 **MODEL LOADING FAILED**")
    st.error(f"Error: {str(e)}")
    
    with st.expander("🔍 View Full Error Details"):
        st.code(traceback.format_exc())
    
    st.warning("⚠️ Cannot proceed without models. Please check the error above.")
    st.stop()

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_with_model(model, img_input, model_name):
    """Make prediction with a single model"""
    try:
        pred = model.predict(img_input, verbose=0)
        non_mri_prob = float(pred[0][0])
        mri_prob = 1.0 - non_mri_prob
        return mri_prob, non_mri_prob
    except Exception as e:
        st.error(f"❌ {model_name} prediction error: {e}")
        return None, None

# ----------------------------
# UI – IMAGE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "📁 Upload an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and models_loaded:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col_info:
        st.write("**Image Information:**")
        st.write(f"📐 Original Size: {image.size}")
        st.write(f"🎨 Mode: {image.mode}")
        st.write(f"📊 Models Available: {len(models)}")
    
    # Preprocess image
    img_input = preprocess(image)
    
    st.markdown("---")
    
    # ----------------------------
    # INDIVIDUAL MODEL PREDICTIONS
    # ----------------------------
    st.header("🤖 Individual Model Predictions")
    
    predictions = {}
    
    with st.spinner("🔍 Running predictions on all models..."):
        time.sleep(0.5)
        
        # Create columns for each model
        cols = st.columns(len(models))
        
        for idx, (model_name, model) in enumerate(models.items()):
            mri_prob, non_mri_prob = predict_with_model(model, img_input, model_name)
            
            if mri_prob is not None:
                predictions[model_name] = {
                    'mri_prob': mri_prob,
                    'non_mri_prob': non_mri_prob
                }
                
                with cols[idx]:
                    st.subheader(model_name)
                    
                    if mri_prob >= 0.5:
                        st.success("🧲 MRI Detected")
                        st.metric("MRI Confidence", f"{mri_prob:.2%}")
                    else:
                        st.error("🚫 NOT MRI")
                        st.metric("Non-MRI Confidence", f"{non_mri_prob:.2%}")
                    
                    # Progress bar
                    st.progress(mri_prob)
                    
                    # Detailed probabilities
                    st.write("**Probabilities:**")
                    st.write(f"MRI: {mri_prob:.4f}")
                    st.write(f"Non-MRI: {non_mri_prob:.4f}")
    
    # ----------------------------
    # ENSEMBLE PREDICTION
    # ----------------------------
    if len(predictions) > 0:
        st.markdown("---")
        st.header("🎯 Ensemble Prediction (Average)")
        
        # Calculate average predictions
        avg_mri_prob = np.mean([p['mri_prob'] for p in predictions.values()])
        avg_non_mri_prob = np.mean([p['non_mri_prob'] for p in predictions.values()])
        
        # Display ensemble result
        ens_col1, ens_col2, ens_col3 = st.columns([2, 1, 1])
        
        with ens_col1:
            if avg_mri_prob >= 0.5:
                st.success("🧲 **ENSEMBLE PREDICTION: MRI IMAGE DETECTED**")
            else:
                st.error("🚫 **ENSEMBLE PREDICTION: NOT AN MRI IMAGE**")
        
        with ens_col2:
            st.metric("MRI Probability", f"{avg_mri_prob:.2%}")
        
        with ens_col3:
            st.metric("Non-MRI Probability", f"{avg_non_mri_prob:.2%}")
        
        # Ensemble progress bar
        st.progress(avg_mri_prob)
        
        # Confidence level indicator
        if avg_mri_prob >= 0.5:
            confidence = avg_mri_prob
            prediction_label = "MRI"
        else:
            confidence = avg_non_mri_prob
            prediction_label = "Non-MRI"
        
        if confidence >= 0.90:
            st.info(f"🎯 **Very High Confidence** - The ensemble is very confident this is {prediction_label}")
        elif confidence >= 0.75:
            st.info(f"✓ **High Confidence** - The ensemble is confident this is {prediction_label}")
        elif confidence >= 0.60:
            st.info(f"⚠️ **Moderate Confidence** - The ensemble suggests this is {prediction_label}")
        else:
            st.warning(f"⚠️ **Low Confidence** - The ensemble is uncertain about this classification")
        
        # ----------------------------
        # MODEL AGREEMENT ANALYSIS
        # ----------------------------
        st.markdown("---")
        st.header("📊 Model Agreement Analysis")
        
        # Count how many models agree
        mri_votes = sum(1 for p in predictions.values() if p['mri_prob'] >= 0.5)
        non_mri_votes = len(predictions) - mri_votes
        
        agree_col1, agree_col2 = st.columns(2)
        
        with agree_col1:
            st.metric("Models Voting MRI", f"{mri_votes}/{len(predictions)}")
            st.metric("Models Voting Non-MRI", f"{non_mri_votes}/{len(predictions)}")
        
        with agree_col2:
            if mri_votes == len(predictions):
                st.success("✅ **Unanimous Agreement** - All models agree: MRI")
            elif non_mri_votes == len(predictions):
                st.success("✅ **Unanimous Agreement** - All models agree: Non-MRI")
            elif mri_votes > non_mri_votes:
                st.info(f"📊 **Majority Agreement** - {mri_votes} models vote MRI")
            elif non_mri_votes > mri_votes:
                st.info(f"📊 **Majority Agreement** - {non_mri_votes} models vote Non-MRI")
            else:
                st.warning("⚠️ **Split Decision** - Models disagree")
        
        # ----------------------------
        # DETAILED COMPARISON TABLE
        # ----------------------------
        st.markdown("---")
        st.header("📋 Detailed Model Comparison")
        
        import pandas as pd
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, preds in predictions.items():
            comparison_data.append({
                'Model': model_name,
                'MRI Probability': f"{preds['mri_prob']:.4f}",
                'Non-MRI Probability': f"{preds['non_mri_prob']:.4f}",
                'Prediction': 'MRI' if preds['mri_prob'] >= 0.5 else 'Non-MRI',
                'Confidence': f"{max(preds['mri_prob'], preds['non_mri_prob']):.2%}"
            })
        
        # Add ensemble row
        comparison_data.append({
            'Model': '**ENSEMBLE (Average)**',
            'MRI Probability': f"{avg_mri_prob:.4f}",
            'Non-MRI Probability': f"{avg_non_mri_prob:.4f}",
            'Prediction': 'MRI' if avg_mri_prob >= 0.5 else 'Non-MRI',
            'Confidence': f"{max(avg_mri_prob, avg_non_mri_prob):.2%}"
        })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # ----------------------------
        # VISUALIZATION
        # ----------------------------
        st.markdown("---")
        st.header("📈 Probability Visualization")
        
        # Create bar chart data
        chart_data = pd.DataFrame({
            'MRI Probability': [p['mri_prob'] for p in predictions.values()],
            'Non-MRI Probability': [p['non_mri_prob'] for p in predictions.values()]
        }, index=list(predictions.keys()))
        
        st.bar_chart(chart_data)
        
        # ----------------------------
        # FINAL SUMMARY
        # ----------------------------
        st.markdown("---")
        st.header("📝 Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**Individual Predictions:**")
            for model_name, preds in predictions.items():
                prediction = "MRI" if preds['mri_prob'] >= 0.5 else "Non-MRI"
                confidence = max(preds['mri_prob'], preds['non_mri_prob'])
                emoji = "🧲" if prediction == "MRI" else "🚫"
                st.write(f"{emoji} {model_name}: {prediction} ({confidence:.2%})")
        
        with summary_col2:
            st.write("**Ensemble Decision:**")
            ensemble_prediction = "MRI" if avg_mri_prob >= 0.5 else "Non-MRI"
            ensemble_confidence = max(avg_mri_prob, avg_non_mri_prob)
            ensemble_emoji = "🧲" if ensemble_prediction == "MRI" else "🚫"
            
            st.write(f"{ensemble_emoji} **{ensemble_prediction}**")
            st.write(f"Confidence: **{ensemble_confidence:.2%}**")
            st.write(f"Agreement: **{max(mri_votes, non_mri_votes)}/{len(predictions)} models**")

# ----------------------------
# SIDEBAR INFO
# ----------------------------
with st.sidebar:
    st.divider()
    st.header("ℹ️ About This App")
    st.write(
        """
        This application uses an ensemble of three deep learning models:
        
        **1. MobileNetV2**
        - Lightweight and fast
        - Efficient for mobile deployment
        
        **2. DenseNet121**
        - Dense connections
        - Good feature reuse
        
        **3. ResNet50**
        - Residual learning
        - Proven reliability
        
        **Ensemble Method:**
        - Averages predictions from all models
        - More robust than single model
        - Reduces individual model errors
        """
    )
    
    st.divider()
    
    st.header("🎯 How It Works")
    st.write("""
    1. **Upload** an image
    2. **Each model** makes an independent prediction
    3. **Ensemble** averages all predictions
    4. **Result** shows individual and combined predictions
    """)
    
    st.divider()
    
    st.header("⚠️ Important Notice")
    st.error(
        """
        **This is a demonstration tool.**
        
        ⚠️ For educational purposes only
        
        ⚠️ Not for medical diagnosis
        
        ⚠️ Always consult healthcare professionals
        """
    )
    
    st.divider()
    
    st.header("📊 Model Statistics")
    if models_loaded and len(models) > 0:
        st.write(f"**Models Loaded:** {len(models)}/{len(MODEL_NAMES)}")
        st.write(f"**Image Size:** {IMG_SIZE[0]}x{IMG_SIZE[1]}")
        st.write(f"**Ensemble Method:** Average")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
    <p>🧠 Ensemble MRI Classification System</p>
    <p>Built with Streamlit & TensorFlow | MobileNetV2 + DenseNet121 + ResNet50</p>
    <p><strong>For Educational & Research Purposes Only</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
