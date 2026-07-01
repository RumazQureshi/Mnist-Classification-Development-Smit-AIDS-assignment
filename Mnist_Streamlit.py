import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import os

# Set page config for a professional look
st.set_page_config(
    page_title="DigitSense AI",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------------------------------
# Neural Network Core Engine (NumPy Implementation)
# ----------------------------------------------------
def run_direct_inference(image_bytes):
    model_file = "model_weights.npz"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model parameters file '{model_file}' not found. Please execute 'python train.py' first.")
    
    # Load neural network parameters (weights & biases)
    network_params = np.load(model_file)
    W1 = network_params["W1"]
    b1 = network_params["b1"]
    W2 = network_params["W2"]
    b2 = network_params["b2"]
    
    # Convert and preprocess uploaded image
    pil_image = Image.open(io.BytesIO(image_bytes))
    grayscale_image = pil_image.convert("L").resize((28, 28))
    normalized_matrix = np.array(grayscale_image) / 255.0
    flattened_vector = normalized_matrix.reshape(1, 784)
    
    # Feedforward Pass
    layer1_z = np.dot(flattened_vector, W1) + b1
    layer1_a = np.maximum(0, layer1_z) # Rectified Linear Unit (ReLU) activation
    layer2_z = np.dot(layer1_a, W2) + b2
    
    # Softmax Activation
    exponentiated_values = np.exp(layer2_z - np.max(layer2_z, axis=1, keepdims=True))
    class_probabilities = exponentiated_values / np.sum(exponentiated_values, axis=1, keepdims=True)
    
    detected_class = int(np.argmax(class_probabilities[0]))
    return detected_class, class_probabilities[0], grayscale_image

# ----------------------------------------------------
# Sidebar Dashboard (Branded Engine Settings)
# ----------------------------------------------------
st.sidebar.title("🧠 DigitSense Controls")
st.sidebar.write("Configure the classifier engine settings below.")

inference_mode = st.sidebar.radio(
    "Classifier Engine",
    options=["⚡ Standalone (Direct Inference)", "🌐 Local API Gateway", "☁️ Remote Cloud Mirror"]
)

api_endpoint = ""
if inference_mode == "🌐 Local API Gateway":
    api_endpoint = st.sidebar.text_input("Gateway Endpoint URL", value="http://localhost:8000/predict")
elif inference_mode == "☁️ Remote Cloud Mirror":
    api_endpoint = st.sidebar.text_input("Cloud Endpoint URL", value="https://ahsanahmede7-imageprediction.hf.space/predict")

# ----------------------------------------------------
# Main UI Layout
# ----------------------------------------------------
# Custom Header with HTML styling
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #4A90E2; font-family: 'Inter', sans-serif;">🧠 DigitSense AI</h1>
        <p style="color: #7F8C8D; font-size: 1.1em;">Neural Digit Interpreter & Vision Visualizer</p>
    </div>
""", unsafe_allow_html=True)

st.write("Upload a handwritten digit (0-9) to analyze it using the Artificial Neural Network.")

# System status message
if inference_mode == "⚡ Standalone (Direct Inference)":
    if os.path.exists("model_weights.npz"):
        st.sidebar.success("🟢 Standalone model parameters loaded successfully.")
    else:
        st.sidebar.error("🔴 'model_weights.npz' parameter file missing!")
        st.warning("⚠️ Local parameter file not found. Please run `python train.py` first, or adjust the Engine Mode in the sidebar.")

st.divider()

user_uploaded_image = st.file_uploader(
    "Drag & drop or browse a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if user_uploaded_image is not None:
    image_bytes = user_uploaded_image.getvalue()
    
    # Preprocess image to display what the AI sees
    try:
        # Load temporary PIL for UI display
        temp_img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    except Exception as e:
        st.error(f"Invalid image format: {e}")
        temp_img = None
        
    if temp_img is not None:
        # Create columns to display original and preprocessed images side-by-side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Raw Upload")
            st.image(user_uploaded_image, width=220)
            
        with col2:
            st.subheader("👁️ AI Vision (28x28)")
            # Scale up the 28x28 image for visibility while preserving pixels
            st.image(temp_img, width=220, output_format="PNG")
        
        st.write("")
        if st.button("Run Digit Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing handwritten digit..."):
                if inference_mode == "⚡ Standalone (Direct Inference)":
                    try:
                        detected_class, prob_distribution, _ = run_direct_inference(image_bytes)
                        
                        st.balloons()
                        st.success(f"### Identified Digit: **{detected_class}**")
                        
                        # Show metric cards
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric(label="Interpretation", value=f"Digit {detected_class}")
                        with metric_col2:
                            st.metric(label="Confidence Score", value=f"{prob_distribution[detected_class] * 100:.2f}%")
                        
                        # Show probabilities chart
                        st.write("📊 **Neural Class Probabilities:**")
                        st.bar_chart(prob_distribution)
                        
                    except Exception as e:
                        st.error(f"Inference Failure: {e}")
                        
                else:
                    # Gateway API calls
                    payload_files = {
                        "file": (
                            user_uploaded_image.name,
                            image_bytes,
                            user_uploaded_image.type
                        )
                    }
                    try:
                        api_response = requests.post(api_endpoint, files=payload_files)
                        if api_response.status_code == 200:
                            response_data = api_response.json()
                            detected_class = response_data["prediction"]
                            
                            st.balloons()
                            st.success(f"### Identified Digit: **{detected_class}**")
                            
                            if "probabilities" in response_data:
                                prob_distribution = response_data["probabilities"]
                                confidence_score = prob_distribution[detected_class] * 100
                                
                                metric_col1, metric_col2 = st.columns(2)
                                with metric_col1:
                                    st.metric(label="Interpretation", value=f"Digit {detected_class}")
                                with metric_col2:
                                    st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
                                
                                st.write("📊 **Neural Class Probabilities:**")
                                st.bar_chart(prob_distribution)
                        else:
                            st.error(f"Gateway Error ({api_response.status_code}): {api_response.text}")
                    except Exception as e:
                        st.error(f"Failed to communicate with the Gateway API: {e}")
                        st.info("Ensure the FastAPI backend server is active (if Local) or verify internet access.")