import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, 
                 num_layers=1, num_classes=7, dropout=0.3, 
                 bidirectional=False, pad_idx=0):
        super(IntentClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 1. Embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embed_dim, 
                                      padding_idx=pad_idx)
        
        # 2. LSTM
        self.lstm = nn.LSTM(input_size=embed_dim, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        
        # 3. Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # 4. Linear
        linear_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(in_features=linear_input_size, out_features=num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        
        # Embedding: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # LSTM
        # out shape: [batch_size, seq_len, hidden_size * num_directions]
        # h_n shape: [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(embedded)
        
        # Extract sentence representation from the last layer
        if self.bidirectional:
            # Concat the final forward hidden state and final backward hidden state
            # h_n[-2] is the forward state of the last layer
            # h_n[-1] is the backward state of the last layer
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]
            
        # Dropout
        dropped = self.dropout(hidden)
        
        # Linear -> logits
        logits = self.fc(dropped)
        
        return logits

def print_model_summary(model):
    print("\\n--- Model Summary ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

if __name__ == "__main__":
    vocab_size = 4000
    max_seq_len = 16
    batch_size = 32
    num_classes = 7
    
    print("Testing Unidirectional Model...")
    model_uni = IntentClassifier(vocab_size=vocab_size, bidirectional=False)
    print_model_summary(model_uni)
    
    # Smoke test unidirectional
    dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    output_uni = model_uni(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_uni.shape}")
    assert output_uni.shape == (batch_size, num_classes), "Unidirectional output shape mismatch!"
    
    print("\\nTesting Bidirectional Model...")
    model_bi = IntentClassifier(vocab_size=vocab_size, bidirectional=True)
    print_model_summary(model_bi)
    
    # Smoke test bidirectional
    output_bi = model_bi(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_bi.shape}")
    assert output_bi.shape == (batch_size, num_classes), "Bidirectional output shape mismatch!"
    
    print("\\nAll smoke tests passed successfully!")
