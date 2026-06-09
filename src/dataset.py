import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import sys

# Import functions from preprocess.py
sys.path.append(os.path.abspath("src"))
from preprocess import clean_text, encode_sentence, load_all_data

class SNIPSDataset(Dataset):
    def __init__(self, data_dir, word2idx, label2idx, max_seq_len):
        self.sentences, self.labels = load_all_data(data_dir)
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.labels[idx]
        
        # Preprocess text
        tokens = clean_text(text)
        encoded_text = encode_sentence(tokens, self.word2idx, self.max_seq_len)
        
        # Preprocess label
        encoded_label = self.label2idx[label]
        
        # Convert to tensors
        text_tensor = torch.tensor(encoded_text, dtype=torch.long)
        label_tensor = torch.tensor(encoded_label, dtype=torch.long)
        
        return text_tensor, label_tensor

def create_dataloaders(train_dir, val_dir, word2idx, label2idx, max_seq_len, batch_size=32):
    train_dataset = SNIPSDataset(train_dir, word2idx, label2idx, max_seq_len)
    val_dataset = SNIPSDataset(val_dir, word2idx, label2idx, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset, val_dataset

if __name__ == "__main__":
    # Load artifacts from Phase 2
    with open("data/processed/word2idx.json", "r") as f:
        word2idx = json.load(f)
    with open("data/processed/idx2label.json", "r") as f:
        idx2label_str = json.load(f)
        idx2label = {int(k): v for k, v in idx2label_str.items()}
    with open("data/processed/metadata.json", "r") as f:
        metadata = json.load(f)
        
    label2idx = {v: k for k, v in idx2label.items()}
    max_seq_len = metadata["max_seq_len"]
    
    train_dir = "data/raw/train"
    val_dir = "data/raw/test"
    
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        train_dir, val_dir, word2idx, label2idx, max_seq_len, batch_size=32
    )
    
    # Verification Checklist tests
    print("\\n--- Dataset Verification ---")
    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"train_dataset[0] types: {type(train_dataset[0][0])}, {type(train_dataset[0][1])}")
    
    # Get a batch
    batch_texts, batch_labels = next(iter(train_loader))
    print(f"\\nBatch text shape: {batch_texts.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    # Decode first 3 sentences
    print("\\n--- First 3 Decoded Sentences from Batch ---")
    idx2word = {idx: word for word, idx in word2idx.items()}
    for i in range(3):
        decoded = [idx2word.get(idx.item(), "<UNK>") for idx in batch_texts[i]]
        # Filter out padding
        decoded = [w for w in decoded if w != "<PAD>"]
        label_name = idx2label[batch_labels[i].item()]
        print(f"Intent: {label_name}")
        print(f"Text: {' '.join(decoded)}\\n")
        
    # Check label range
    all_labels_valid = all(0 <= label.item() <= 6 for label in batch_labels)
    print(f"Label values are integers 0-6: {all_labels_valid}")
    print("Dataset implementation successful!")
