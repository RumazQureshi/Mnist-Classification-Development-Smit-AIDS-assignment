import os
import gzip
import urllib.request
import numpy as np

# Configuration
MNIST_DIR = "mnist_data"
WEIGHTS_FILE = "model_weights.npz"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "X_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "X_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz"
}

def download_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)
    for name, filename in FILES.items():
        filepath = os.path.join(MNIST_DIR, filename)
        if not os.path.exists(filepath):
            url = BASE_URL + filename
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded {filename}.")

def load_images(filepath):
    print(f"Loading images from {filepath}...")
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype('>i4'))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols) / 255.0

def load_labels(filepath):
    print(f"Loading labels from {filepath}...")
    with gzip.open(filepath, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype=np.dtype('>i4'))
        return np.frombuffer(f.read(), dtype=np.uint8)

def train():
    download_mnist()
    
    # Load dataset
    X_train = load_images(os.path.join(MNIST_DIR, FILES["X_train"]))
    y_train = load_labels(os.path.join(MNIST_DIR, FILES["y_train"]))
    X_test = load_images(os.path.join(MNIST_DIR, FILES["X_test"]))
    y_test = load_labels(os.path.join(MNIST_DIR, FILES["y_test"]))
    
    print(f"Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set: X={X_test.shape}, y={y_test.shape}")
    
    # Network dimensions
    input_size = 784
    hidden_size = 512
    output_size = 10
    
    # He initialization for W1, Xavier for W2
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    
    # Adam parameters
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    
    t = 0
    epochs = 10
    batch_size = 128
    num_samples = X_train.shape[0]
    
    print("\nStarting ANN training in NumPy (Adam Optimizer)...")
    for epoch in range(epochs):
        # Shuffle training set
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_s = X_train[indices]
        y_train_s = y_train[indices]
        
        epoch_loss = 0.0
        correct_train = 0
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train_s[i : i + batch_size]
            y_batch = y_train_s[i : i + batch_size]
            m = X_batch.shape[0]
            
            # Forward pass
            z1 = np.dot(X_batch, W1) + b1
            a1 = np.maximum(0, z1) # ReLU
            
            # 20% Dropout (training only)
            drop_mask = (np.random.rand(*a1.shape) >= 0.2) / 0.8
            a1_drop = a1 * drop_mask
            
            z2 = np.dot(a1_drop, W2) + b2
            
            # Softmax
            exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            a2 = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
            
            # Categorical cross entropy loss
            y_onehot = np.zeros((m, output_size))
            y_onehot[np.arange(m), y_batch] = 1.0
            
            loss = -np.sum(y_onehot * np.log(a2 + 1e-15)) / m
            epoch_loss += loss * m
            
            correct_train += np.sum(np.argmax(a2, axis=1) == y_batch)
            
            # Backward pass
            dz2 = (a2 - y_onehot) / m
            dW2 = np.dot(a1_drop.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, W2.T)
            dz1 = da1 * (z1 > 0) * drop_mask
            
            dW1 = np.dot(X_batch.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Adam step updates
            t += 1
            
            # Update layer 1
            mW1 = beta1 * mW1 + (1 - beta1) * dW1
            vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
            mW1_h = mW1 / (1 - beta1 ** t)
            vW1_h = vW1 / (1 - beta2 ** t)
            W1 -= learning_rate * mW1_h / (np.sqrt(vW1_h) + eps)
            
            mb1 = beta1 * mb1 + (1 - beta1) * db1
            vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
            mb1_h = mb1 / (1 - beta1 ** t)
            vb1_h = vb1 / (1 - beta2 ** t)
            b1 -= learning_rate * mb1_h / (np.sqrt(vb1_h) + eps)
            
            # Update layer 2
            mW2 = beta1 * mW2 + (1 - beta1) * dW2
            vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
            mW2_h = mW2 / (1 - beta1 ** t)
            vW2_h = vW2 / (1 - beta2 ** t)
            W2 -= learning_rate * mW2_h / (np.sqrt(vW2_h) + eps)
            
            mb2 = beta1 * mb2 + (1 - beta1) * db2
            vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
            mb2_h = mb2 / (1 - beta1 ** t)
            vb2_h = vb2 / (1 - beta2 ** t)
            b2 -= learning_rate * mb2_h / (np.sqrt(vb2_h) + eps)
            
        # Calculate training epoch metrics
        train_loss = epoch_loss / num_samples
        train_acc = correct_train / num_samples
        
        # Test set evaluation
        z1_t = np.dot(X_test, W1) + b1
        a1_t = np.maximum(0, z1_t)
        z2_t = np.dot(a1_t, W2) + b2
        exp_z2_t = np.exp(z2_t - np.max(z2_t, axis=1, keepdims=True))
        a2_t = exp_z2_t / np.sum(exp_z2_t, axis=1, keepdims=True)
        test_acc = np.mean(np.argmax(a2_t, axis=1) == y_test)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
    # Save the final model weights
    np.savez(WEIGHTS_FILE, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"\nWeights saved successfully to '{WEIGHTS_FILE}'.")

if __name__ == "__main__":
    train()
