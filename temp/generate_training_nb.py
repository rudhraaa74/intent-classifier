import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Setup cell
cell_setup = """import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Setup paths
sys.path.append(os.path.abspath("../src"))
from dataset import create_dataloaders
from model import IntentClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
"""

# Config cell
cell_config = """# Define configuration
config = {
  "seed": 42,
  "embed_dim": 128,
  "hidden_size": 256,
  "num_layers": 1,
  "bidirectional": True,
  "dropout": 0.3,
  "batch_size": 32,
  "learning_rate": 0.001,
  "max_epochs": 30,
  "early_stopping_patience": 5,
}

# Set seed
torch.manual_seed(config["seed"])

# Load metadata from Phase 2
with open("../data/processed/metadata.json", "r") as f:
    metadata = json.load(f)
    
config["vocab_size"] = metadata["vocab_size"]
config["max_seq_len"] = metadata["max_seq_len"]

# Save config
exp_dir = "../experiments/exp_002_bidirectional"
os.makedirs(exp_dir, exist_ok=True)
with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("Config saved!")
"""

# Dataloaders cell
cell_data = """# Load Data
with open("../data/processed/word2idx.json", "r") as f:
    word2idx = json.load(f)
with open("../data/processed/idx2label.json", "r") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}
label2idx = {v: k for k, v in idx2label.items()}

train_loader, val_loader, _, _ = create_dataloaders(
    train_dir="../data/raw/train",
    val_dir="../data/raw/test",
    word2idx=word2idx,
    label2idx=label2idx,
    max_seq_len=config["max_seq_len"],
    batch_size=config["batch_size"]
)
print("DataLoaders ready.")
"""

# Model cell
cell_model = """# Initialize model
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

print(model)
"""

# Training loop
cell_train = """# Training Loop
best_val_accuracy = 0.0
epochs_no_improve = 0

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

os.makedirs("../models", exist_ok=True)

for epoch in range(1, config["max_epochs"] + 1):
    # --- TRAINING ---
    model.train()
    total_loss, correct, total = 0, 0, 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    
    # --- VALIDATION ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            logits = model(texts)
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    
    # Update scheduler
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch}/{config['max_epochs']} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    # Checkpointing
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        epochs_no_improve = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_accuracy': val_acc,
            'hyperparameters': config
        }, "../models/best_model.pt")
        print("  -> Saved new best model!")
    else:
        epochs_no_improve += 1
        
    # Early Stopping
    if epochs_no_improve >= config["early_stopping_patience"]:
        print(f"Early stopping triggered at epoch {epoch}")
        break

print(f"\\nTraining complete! Best Validation Accuracy: {best_val_accuracy*100:.2f}%")
"""

# Plotting cell
cell_plot = """# Plotting
import matplotlib.pyplot as plt

os.makedirs("../assets", exist_ok=True)

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label='Train Loss')
plt.plot(epochs, history["val_loss"], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_acc"], label='Train Acc')
plt.plot(epochs, history["val_acc"], label='Validation Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("../assets/training_curves.png")
plt.show()

# Save history
pd.DataFrame(history).to_csv(os.path.join(exp_dir, "training_log.csv"), index=False)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Phase 5 — Training Loop"),
    nbf.v4.new_code_cell(cell_setup),
    nbf.v4.new_code_cell(cell_config),
    nbf.v4.new_code_cell(cell_data),
    nbf.v4.new_code_cell(cell_model),
    nbf.v4.new_code_cell(cell_train),
    nbf.v4.new_code_cell(cell_plot)
]

os.makedirs("notebooks", exist_ok=True)
with open('notebooks/02_training_bidirectional.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated!")
