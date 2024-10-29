import streamlit as st
import torch
import pickle
from torch import nn

# Redefine the NextWord class in the current script
class NextWord(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)  # shape: (batch_size, block_size, emb_dim)
        x, _ = self.rnn(x)  # shape: (batch_size, block_size, hidden_size)
        x = x[:, -1, :]  # take the output of the last time step
        x = self.lin(x)  # shape: (batch_size, vocab_size)
        return x

# Load model and vocab mappings
model = torch.load('next_word_model.pth', map_location=torch.device('cpu'))
with open('vocab_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

stoi, itos = mappings['stoi'], mappings['itos']
device = torch.device("cpu")

# Streamlit UI for parameter input
st.title("Next-Word Prediction App")
context_length = st.sidebar.slider("Context Length", 2, 10, 5)
num_words = st.sidebar.number_input("Number of Words to Generate", 1, 50, 20)
start_text = st.text_input("Enter starting text:", value="I had seen little")

def generate_text(model, itos, block_size, start_text, max_len):
    start_words = start_text.lower().split()
    context = ([0] * (block_size - len(start_words)) + [stoi.get(word, 0) for word in start_words]
               if len(start_words) < block_size else
               [stoi.get(word, 0) for word in start_words[-block_size:]])

    generated = []
    for _ in range(max_len):
        x = torch.tensor(context).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(x)
        next_word_ix = torch.argmax(y_pred, dim=1).item()
        generated.append(itos.get(next_word_ix, ''))
        context = context[1:] + [next_word_ix]
    
    return ' '.join(generated)

if st.button("Generate Text"):
    st.write(f"**Generated Text:** {start_text} {generate_text(model, itos, context_length, start_text, num_words)}")
