import os
import sys
import json
import torch

# Ensure we can import from src if run from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text, encode_sentence
from model import IntentClassifier

class Predictor:
    def __init__(self, 
                 model_path="models/best_model.pt", 
                 word2idx_path="data/processed/word2idx.json",
                 idx2label_path="data/processed/idx2label.json",
                 config_path="experiments/exp_002_bidirectional/config.json"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load configs and maps
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        with open(word2idx_path, "r") as f:
            self.word2idx = json.load(f)
            
        with open(idx2label_path, "r") as f:
            self.idx2label = {int(k): v for k, v in json.load(f).items()}
            
        # Load Model
        self.model = IntentClassifier(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            num_classes=len(self.idx2label),
            dropout=self.config["dropout"],
            bidirectional=self.config["bidirectional"],
            pad_idx=self.word2idx.get("<PAD>", 0)
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, sentence):
        # Preprocess
        tokens = clean_text(sentence)
        encoded = encode_sentence(tokens, self.word2idx, self.config["max_seq_len"])
        
        # Convert to tensor (batch size 1)
        tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        # Get highest confidence intent
        conf, pred_idx = torch.max(probs, dim=0)
        intent = self.idx2label[pred_idx.item()]
        
        # Map all scores
        all_scores = {self.idx2label[i]: probs[i].item() for i in range(len(self.idx2label))}
        
        return {
            "intent": intent,
            "confidence": conf.item(),
            "all_scores": all_scores
        }


if __name__ == "__main__":
    # Ensure it's run from the root of the project
    if not os.path.exists("models/best_model.pt"):
        print("Error: Please run this script from the project root directory: python src/predict.py")
        sys.exit(1)
        
    predictor = Predictor()
    
    print("=== Intent Classifier ===")
    print("Type a sentence. Press Ctrl+C to exit.\\n")
    
    try:
        while True:
            sentence = input("You: ")
            if not sentence.strip():
                continue
                
            result = predictor.predict(sentence)
            print(f"Intent:     {result['intent']}")
            print(f"Confidence: {result['confidence'] * 100:.2f}%")
            print("") # Newline for spacing
            
    except KeyboardInterrupt:
        print("\nExiting. Have a great day!")
