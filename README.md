# Intent Classification Chatbot

This project implements a custom **Bidirectional LSTM** from scratch in PyTorch to classify natural language text into 7 distinct intents using the popular SNIPS dataset.

## 🚀 Performance
The final Bidirectional LSTM model achieved an impressive **98.43% Validation Accuracy** on the unseen test set!

![Confusion Matrix](assets/confusion_matrix.png)

## 📂 Project Structure

Following strict separation of concerns, the project is structured as follows:

- **`data/`**: Raw JSON data from SNIPS, alongside the processed vocabulary and label maps.
- **`src/`**: Pure, finalized Python source code. This includes the preprocessing pipeline (`preprocess.py`), PyTorch DataLoader generation (`dataset.py`), and the model architecture (`model.py`).
- **`notebooks/`**: Interactive execution environments. Training, evaluation, and inference all live here so the outputs are documented and visually verifiable.
- **`experiments/`**: Every distinct hyperparameter configuration gets its own isolated folder to prevent data loss. The champion run is `exp_002_bidirectional`.
- **`models/`**: The saved `best_model.pt` PyTorch weights.

## 🏃 How to Run

1. **Setup your environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Explore the data:**
   Open `notebooks/01_exploration.ipynb` to see the dataset distribution.

3. **Train the model:**
   Open `notebooks/02_training.ipynb` and run all cells. This handles data loading, model initialization, and the training loop with Early Stopping and a Learning Rate Scheduler.

4. **Evaluate:**
   Open `notebooks/03_results_analysis.ipynb` to generate the classification report and confusion matrix.

5. **Test it yourself!**
   Open `notebooks/04_inference.ipynb` and run the interactive loop. You can type any custom sentence to see the live intent prediction!

---

## ⚠️ Limitations

While the Bidirectional LSTM achieved fantastic accuracy on SNIPS, the architecture does have inherent limitations:

1. **Context Length Constraint:** LSTMs struggle to maintain context over very long sequences due to the bottleneck of squishing all sequence history into a single hidden state vector. We mitigated this by setting a strict `max_seq_len` of 16, but this means long inputs are aggressively truncated.
2. **Sequential Processing:** Because LSTMs process tokens one-by-one, they cannot easily parallelize training across GPUs like modern architectures.
3. **Out-of-Vocabulary Tokens:** We capped our vocabulary at 4,000 words. Any word outside of this (like a rare name or misspelled city) is immediately replaced with `<UNK>`, stripping it of all semantic meaning.

## 🌉 The Transformer Bridge (Future Work)

To overcome the limitations above, the natural next step for this project would be to bridge over to a **Transformer-based architecture** (such as BERT or RoBERTa). 

Instead of training word embeddings from scratch on a tiny dataset, we could:
1. **Leverage Pre-trained Context:** Replace the custom `<UNK>`-prone `word2idx` vocabulary with a pre-trained subword tokenizer (like WordPiece).
2. **Global Attention:** Replace the LSTM layers with Self-Attention layers, allowing the model to look at the entire sequence simultaneously without the context degradation over time.
3. **Fine-Tuning:** By loading a pre-trained `distilbert-base-uncased` model and slapping a simple Linear classification head on top, we could likely match or exceed this 98% accuracy while being highly robust to misspellings and complex sentence structures!
