import streamlit as st
import torch
import torch.nn as nn
import os
import random

# Define the model class
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, block_size, activation='relu'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim * block_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
        # Set the activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Choose from 'relu' or 'tanh'.")

    def forward(self, x):
        x = self.emb(x)  # Shape: (batch_size, block_size, emb_dim)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, block_size * emb_dim)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Sidebar for model selection and parameters
st.sidebar.title("Model and Prediction Settings")

# User-selectable parameters
block_size = st.sidebar.selectbox("Context Length (Block Size)", [5, 10, 15], index=2)
embedding_dim = st.sidebar.selectbox("Embedding Dimension", [64, 128], index=1)
activation = st.sidebar.selectbox("Activation Function", ['relu', 'tanh'], index=0)
num_predictions = st.sidebar.slider("Number of Words to Predict", 1, 50, value=5)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Set random seed for reproducibility
torch.manual_seed(random_seed)
random.seed(random_seed)

# Define the model filename based on user selection
model_filename = f"trained_model_{block_size}_{embedding_dim}_{activation}.pth"

# Load model function
def load_model(filename):
    # Load the saved state dict
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    vocab_size = len(checkpoint['stoi']) + 1
    model = NextWordMLP(vocab_size, checkpoint['emb_dim'], 128, checkpoint['block_size'], checkpoint['activation'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['stoi'], checkpoint['itos']

# Load the selected model
try:
    model, stoi, itos = load_model(model_filename)
    st.write(f"Loaded model: {model_filename}")
except FileNotFoundError:
    st.error("The selected model file was not found. Please upload the correct .pth files.")

# Input text box for prediction
input_text = st.text_input("Enter some text to start the prediction:")

# Prediction function for generating multiple words
def predict_next_words(model, input_text, num_predictions, block_size):
    words = input_text.strip().split()
    # Ensure the input text matches the block size (pad or truncate as needed)
    input_words = words[-block_size:]
    unk_index = stoi.get('<unk>', 0)
    input_indices = [stoi.get(word, unk_index) for word in input_words]
    
    if len(input_indices) < block_size:
        input_indices = [0] * (block_size - len(input_indices)) + input_indices
    elif len(input_indices) > block_size:
        input_indices = input_indices[-block_size:]

    predictions = []
    for _ in range(num_predictions):
        input_tensor = torch.tensor([input_indices])
        with torch.no_grad():
            output = model(input_tensor)
            predicted_index = output.argmax(dim=1).item()
            predicted_word = itos[predicted_index]
            predictions.append(predicted_word)
            
            # Update input indices for the next prediction
            input_indices = input_indices[1:] + [predicted_index]

    return predictions

# Generate predictions if the button is clicked
if st.button("Predict Next Words"):
    if input_text:
        predictions = predict_next_words(model, input_text, num_predictions, block_size)
        st.write("Predicted text: " + " ".join(predictions))
    else:
        st.warning("Please enter some text for prediction.")