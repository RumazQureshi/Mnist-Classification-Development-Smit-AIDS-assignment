# 🧠 DigitSense AI — MNIST Digit Classification

DigitSense AI is a lightweight, modern, and interactive handwritten digit classifier built from scratch. It uses a custom **Artificial Neural Network (ANN)** trained on the MNIST dataset and offers a feature-rich web application interface.

## ✨ Features
- **Pure NumPy Inference Engine**: Run prediction models directly in the web app or API using raw NumPy matrices. No heavy TensorFlow or PyTorch dependencies are required at runtime.
- **Top-Tier Performance**: The 3-layer neural network achieves **99.36% training accuracy** and **98.20% test accuracy**.
- **Interactive UI**: Upload any handwritten digit image and view the classification results instantly.
- **AI Vision Visualizer**: Displays the raw upload next to the processed $28 \times 28$ grayscale matrix, showing exactly how the AI processes and "sees" your digit.
- **Metrics & Data Charts**: Real-time confidence scores and probability distribution bar charts for digits 0-9.
- **FastAPI Backend**: A complete local FastAPI gateway server is included to run predictions via API requests.

---

## 📁 Project Structure
- **`Mnist_Streamlit.py`**: The main Streamlit web application.
- **`model_weights.npz`**: Pre-trained model parameter weights (3.2 MB).
- **`train.py`**: Automated script to train the model from scratch in NumPy using the Adam optimizer.
- **`Backend/imagePrediction/main.py`**: A FastAPI application that hosts the model and exposes a POST `/predict` API.
- **`requirements.txt`**: Python dependencies required to run the frontend and API.
- **`Seven.png`**: Sample image for quick testing.

---

## 🚀 Local Setup & Execution

### 1. Installation
Clone the repository and install the dependencies in a Python virtual environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\Activate.ps1

# Activate it (Mac/Linux)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Streamlit Application (Recommended)
Launch the interactive web app locally:
```bash
streamlit run Mnist_Streamlit.py
```
Open `http://localhost:8501` in your browser. Choose **Standalone Mode** in the sidebar to run predictions instantly.

### 3. Run the FastAPI Backend (Optional)
Start the FastAPI server:
```bash
uvicorn Backend.imagePrediction.main:app --reload
```
The API documentation will be available at `http://localhost:8000/docs`. You can route your Streamlit app through this API by choosing **FastAPI Local Server** in the sidebar settings.

### 4. Training from Scratch (Optional)
If you wish to retrain the network and update the weights file, simply execute:
```bash
python train.py
```
The script will download the raw MNIST dataset files, run 10 epochs of mini-batch gradient descent (Adam), and overwrite `model_weights.npz` with the updated parameters.

---

## ☁️ Deployment Guide (Get a Free Live Link)

To submit a live URL of your project, follow these instructions to host the Streamlit app for free:

### Option A: Streamlit Community Cloud (Recommended)
1. Upload your code files (`Mnist_Streamlit.py`, `model_weights.npz`, `requirements.txt`, `Seven.png`) to your GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **Create app** (or **New app**), select your repository, specify the branch (`main`), and set the main file path to `Mnist_Streamlit.py`.
4. Click **Deploy!** Your app will be live in 1–2 minutes with a shareable URL.

### Option B: Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) and select **Streamlit** as the SDK.
2. Upload `Mnist_Streamlit.py` (rename it to `app.py` or edit the space settings entrypoint), `model_weights.npz`, `requirements.txt`, and `Seven.png` to the repository.
3. Hugging Face will build and launch your web app automatically.
