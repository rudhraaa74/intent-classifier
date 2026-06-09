import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Setup cell
cell_setup = """import os
import json
import torch
import sys

# Setup paths
sys.path.append(os.path.abspath("../src"))
from preprocess import clean_text, encode_sentence
from model import IntentClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
"""

# Load cell
cell_load = """# Load Checkpoint and Config
checkpoint = torch.load("../models/best_model.pt", map_location=device, weights_only=False)
config = checkpoint["hyperparameters"]

# Load Data maps
with open("../data/processed/word2idx.json", "r") as f:
    word2idx = json.load(f)
with open("../data/processed/idx2label.json", "r") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}
label2idx = {v: k for k, v in idx2label.items()}

# Initialize model
model = IntentClassifier(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    num_classes=len(label2idx),
    dropout=config["dropout"],
    bidirectional=config["bidirectional"],
    pad_idx=word2idx.get("<PAD>", 0)
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model and dictionaries loaded successfully!")
"""

# Predict function cell
cell_func = """def predict_intent(sentence):
    # Preprocess
    tokens = clean_text(sentence)
    encoded = encode_sentence(tokens, word2idx, config["max_seq_len"])
    
    # Convert to tensor
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        
    intent = idx2label[pred_idx.item()]
    confidence = conf.item() * 100
    
    print(f"Input:      \\"{sentence}\\"")
    print(f"Predicted:  {intent}")
    print(f"Confidence: {confidence:.2f}%\\n")
    
    return intent, confidence
"""

# Interactive Test cell
cell_test = """# Try your own sentences here!
predict_intent("play the new album by taylor swift")
predict_intent("what is the weather like in tokyo tomorrow")
predict_intent("add this song to my workout playlist")
predict_intent("book a table for 4 at the italian restaurant tonight")
predict_intent("find me showtimes for the new marvel movie")
predict_intent("rate this book 5 out of 5 stars")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Phase 7 — Interactive Inference"),
    nbf.v4.new_code_cell(cell_setup),
    nbf.v4.new_code_cell(cell_load),
    nbf.v4.new_code_cell(cell_func),
    nbf.v4.new_code_cell(cell_test)
]

os.makedirs("notebooks", exist_ok=True)
with open('notebooks/04_inference.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated!")
