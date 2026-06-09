import os
import json
import re
from collections import Counter

def clean_text(sentence):
    """Lowercases, removes punctuation, strips extra whitespace, returns list of tokens."""
    sentence = sentence.lower()
    # Remove punctuation using regex (keeps letters and numbers)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Strip and split by whitespace
    return sentence.split()

def build_vocab(list_of_tokenized_sentences, max_vocab_size):
    """Builds vocabulary with <PAD> and <UNK>."""
    # Count frequencies
    all_tokens = [token for tokens in list_of_tokenized_sentences for token in tokens]
    word_counts = Counter(all_tokens)
    
    # Keep top max_vocab_size words (subtract 2 for PAD and UNK)
    most_common = word_counts.most_common(max_vocab_size - 2)
    
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx
        
    return word2idx

def encode_sentence(tokens, word2idx, max_seq_len):
    """Converts tokens to indices, applying padding/truncation."""
    indices = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    
    if len(indices) < max_seq_len:
        # Pad on the right
        indices.extend([word2idx["<PAD>"]] * (max_seq_len - len(indices)))
    else:
        # Truncate
        indices = indices[:max_seq_len]
        
    return indices

def build_label_encoder(list_of_labels):
    """Assigns unique integer to each intent class."""
    unique_labels = sorted(list(set(list_of_labels)))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

def load_all_data(split_dir):
    import glob
    files = glob.glob(os.path.join(split_dir, "*.json"))
    sentences = []
    labels = []
    for f in files:
        with open(f, 'r', encoding='utf-8', errors='replace') as file:
            data = json.load(file)
            for intent, examples in data.items():
                for example in examples:
                    text = "".join([chunk.get("text", "") for chunk in example.get("data", [])]).strip()
                    sentences.append(text)
                    labels.append(intent)
    return sentences, labels

if __name__ == "__main__":
    # Hyperparameters from Phase 1
    max_seq_len = 16
    max_vocab_size = 4000
    
    # Load training data
    print("Loading training data...")
    train_sentences, train_labels = load_all_data("data/raw/train")
    
    # Step 1: Clean text
    print("Cleaning text...")
    train_tokenized = [clean_text(s) for s in train_sentences]
    
    # Step 2: Build vocabulary
    print("Building vocabulary...")
    word2idx = build_vocab(train_tokenized, max_vocab_size)
    
    # Step 4: Build label encoder
    print("Building label encoder...")
    label2idx, idx2label = build_label_encoder(train_labels)
    
    # Step 5: Save artifacts
    print("Saving artifacts...")
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/word2idx.json", "w") as f:
        json.dump(word2idx, f, indent=2)
    with open("data/processed/idx2label.json", "w") as f:
        json.dump(idx2label, f, indent=2)
        
    metadata = {
        "max_seq_len": max_seq_len,
        "vocab_size": len(word2idx)
    }
    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Step 6: Sanity check
    print("\\n--- Sanity Check ---")
    test_sentences = [
        "What's the weather like in New York?",
        "Play some jazz music by Miles Davis.",
        "Book a restaurant for two at 8 PM.",
        "Add this track to my indie playlist.",
        "Find a screening for the new Marvel movie."
    ]
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    for s in test_sentences:
        tokens = clean_text(s)
        encoded = encode_sentence(tokens, word2idx, max_seq_len)
        decoded = [idx2word.get(idx, "<UNK>") for idx in encoded]
        
        print(f"Original: {s}")
        print(f"Tokens:   {tokens}")
        print(f"Encoded:  {encoded}")
        print(f"Decoded:  {decoded}")
        print("-" * 40)
        
    # Sanity check verifications
    print(f"word2idx['<PAD>'] == 0: {word2idx.get('<PAD>') == 0}")
    print(f"word2idx['<UNK>'] == 1: {word2idx.get('<UNK>') == 1}")
    print(f"Unknown word mapping: {encode_sentence(['supercalifragilistic'], word2idx, 5)[0] == 1}")
    print(f"Label encoder covers 7 classes: {len(label2idx) == 7}")
    print(f"max_seq_len check: {len(encode_sentence(['test'], word2idx, max_seq_len)) == max_seq_len}")
    print("Preprocessing successful!")
