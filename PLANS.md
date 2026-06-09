# Intent Classification Chatbot — Project Plan
### LSTM on the SNIPS Dataset | PyTorch

---

> **How to use this document**
> Each phase below is self-contained. Before any code for a phase is written, you review
> the plan for that phase and give explicit approval. No phase begins until the previous
> one is approved and verified. Checkboxes at the end of each phase list exactly what
> you should verify before sign-off.

---

## Phase 0 — Environment Setup

**Goal:** Get a clean, reproducible Python environment with every dependency installed
before writing a single line of project code.

### What will be done

1. Create a project folder with the following structure:

```
intent-classifier/
├── data/
│   ├── raw/                ← raw dataset goes here
│   │   ├── train/
│   │   └── test/
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── checkpoints/            ← saved model weights go here
├── notebooks/
│   └── 01_exploration.ipynb
├── requirements.txt
└── README.md
```

2. Create a `requirements.txt` with pinned versions:

```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
tqdm>=4.65.0
```

3. Set up a virtual environment and install all dependencies.

4. Verify PyTorch is installed correctly and can detect any available hardware
   (CPU is perfectly fine for this project — SNIPS is a small dataset).

5. Download the SNIPS dataset from GitHub. The dataset lives at:
   `https://github.com/sonos/nlu-benchmark`
   It is structured as JSON files, one folder per intent, each containing
   `train_<IntentName>.json` and `validate_<IntentName>.json` files.

### Why this phase exists

Skipping environment setup is the #1 cause of debugging time loss. Doing it
explicitly as its own phase ensures everything downstream runs without
dependency surprises.

### Verification checklist before approving Phase 1

- [ ] Project folder structure exists as shown above
- [ ] `pip install -r requirements.txt` runs without errors
- [ ] `import torch; print(torch.__version__)` works in Python
- [ ] SNIPS JSON files are downloaded and visible inside `data/snips/`
- [ ] You can launch a Jupyter notebook from the project root

---

## Phase 1 — Data Exploration

**Goal:** Deeply understand the dataset before making any modeling decisions.
Every architectural choice in later phases will be justified by what we find here.

### What will be done

All work in this phase happens inside `notebooks/01_exploration.ipynb`.

**Step 1 — Load the raw JSON files**

Parse all training and validation JSON files across all 7 intents and assemble
them into a flat structure: a list of (utterance_text, intent_label) pairs.

**Step 2 — Class distribution analysis**

Count utterances per intent in both the train and validation splits.
Plot a bar chart. This tells us whether the dataset is balanced or skewed.
If one intent has 10x more examples than another, the model will be biased
toward predicting it — and we need to know this upfront.

The 7 intents are:
- PlayMusic
- GetWeather
- BookRestaurant
- RateBook
- AddToPlaylist
- SearchCreativeWork
- SearchScreeningEvent

**Step 3 — Sentence length analysis**

For every utterance, count the number of words. Then:
- Plot a histogram of sentence lengths
- Compute min, max, mean, median, and 95th percentile length

This directly determines what `max_seq_len` to use when padding sequences later.
Setting it too short cuts off information; too long wastes computation and
dilutes the LSTM's memory.

**Step 4 — Vocabulary analysis**

Tokenize all training utterances (split by whitespace, lowercase). Count
unique words and word frequencies. Plot a frequency distribution.

This tells us:
- How large our vocabulary is
- How many words appear only once or twice (rare words that can be replaced
  with `<UNK>` without losing much information)
- What a sensible vocabulary cap (top-N words) should be

**Step 5 — Sample inspection**

Print 10 random examples from each intent side by side. The goal is to get
an intuitive feel for:
- How much vocabulary overlap there is between intents
- Which intents seem easy to separate vs potentially confusable
- Whether the text needs heavy cleaning or is already fairly clean

**Step 6 — Key decisions to lock in**

By the end of this phase, we will have confirmed:

| Decision | What we look at | Likely value |
|---|---|---|
| `max_seq_len` | 95th percentile of sentence lengths | ~15–20 |
| `vocab_size` | frequency distribution cutoff | ~3000–5000 |
| Class balance strategy | bar chart of intent counts | likely none needed |

### Why this phase matters

Every number we hardcode in later phases — padding length, vocabulary size,
embedding size — should be traceable back to something we observed here.
Guessing these values is how you end up with a model that silently performs
poorly for non-obvious reasons.

### Verification checklist before approving Phase 2

- [ ] Bar chart of utterances per intent is visible and makes sense
- [ ] Sentence length histogram is visible; 95th percentile is noted
- [ ] You know the total vocabulary size of the training set
- [ ] You've read at least 5 examples from each intent class
- [ ] `max_seq_len` and `vocab_size` are agreed upon and written down

---

## Phase 2 — Text Preprocessing Pipeline

**Goal:** Convert raw text strings into clean integer sequences that the model
can consume. Build this as a reusable, self-contained pipeline.

### What will be done

All work in this phase lives in `src/preprocess.py`.

**Step 1 — Text cleaning function**

Write a `clean_text(sentence)` function that:
- Lowercases the string
- Removes punctuation (commas, periods, question marks, etc.)
- Strips extra whitespace
- Returns a list of tokens (words)

Example:
```
Input:  "What's the weather in Bangalore?"
Output: ["whats", "the", "weather", "in", "bangalore"]
```

**Step 2 — Vocabulary builder**

Write a `build_vocab(list_of_tokenized_sentences, max_vocab_size)` function that:
- Counts word frequencies across all training sentences
- Keeps only the top `max_vocab_size` most frequent words
- Assigns integer indices starting from 2 (reserving 0 and 1)
- Explicitly adds two special tokens:
  - `<PAD>` → index 0 (used to fill short sentences to uniform length)
  - `<UNK>` → index 1 (used for words not in the vocabulary at test time)
- Returns a `word2idx` dictionary: `{"<PAD>": 0, "<UNK>": 1, "play": 2, ...}`

**Critical rule:** The vocabulary is built from the training set ONLY. The
validation and test sets may contain words not in the vocabulary — those
become `<UNK>`. This simulates real-world deployment where the user can say
anything.

**Step 3 — Sentence encoder**

Write an `encode_sentence(tokens, word2idx, max_seq_len)` function that:
- Converts a list of tokens to a list of integers using `word2idx`
- Maps any unknown token to index 1 (`<UNK>`)
- If the sentence is shorter than `max_seq_len`, pads with 0s on the right
- If the sentence is longer than `max_seq_len`, truncates to `max_seq_len`
- Returns a fixed-length list of integers

Example (max_seq_len = 7):
```
Tokens:  ["play", "some", "jazz", "music"]
Encoded: [2,      45,     305,    89,    0, 0, 0]
```

**Step 4 — Label encoder**

Write a `build_label_encoder(list_of_labels)` function that:
- Assigns a unique integer to each intent class
- Returns both `label2idx` and `idx2label` dictionaries

Example:
```
label2idx = {"PlayMusic": 0, "GetWeather": 1, "BookRestaurant": 2, ...}
idx2label = {0: "PlayMusic", 1: "GetWeather", 2: "BookRestaurant", ...}
```

**Step 5 — Save artifacts**

Save `word2idx`, `idx2label`, and the confirmed `max_seq_len` to disk
(as JSON or pickle). These are needed at inference time — without them,
you cannot decode the model's predictions back into intent names.

**Step 6 — Sanity check**

Write a short test script that:
- Encodes 5 sample sentences
- Decodes them back to tokens using `idx2word`
- Verifies the round-trip is correct and padding/truncation works as expected

### Why this phase matters

The preprocessing pipeline is the foundation. If there is a bug here —
a vocabulary built on the wrong split, an off-by-one in padding, a label
mapping that's inconsistent — the model will train on corrupted data and
produce silently wrong results. Verifying this phase independently before
training is critical.

### Verification checklist before approving Phase 3

- [ ] `clean_text("What's the weather?")` returns clean tokens
- [ ] `word2idx` has `<PAD>` at index 0 and `<UNK>` at index 1
- [ ] An unknown word maps to index 1, not an error
- [ ] `encode_sentence` always returns a list of exactly `max_seq_len` integers
- [ ] Label encoder covers all 7 intent classes
- [ ] Round-trip sanity check passes for at least 5 sentences
- [ ] `word2idx` and `idx2label` are saved to disk

---

## Phase 3 — Dataset and DataLoader

**Goal:** Wrap the preprocessed data in PyTorch's data loading infrastructure
so the model receives clean batches of tensors during training.

### What will be done

All work in this phase lives in `src/dataset.py`.

**Step 1 — Custom Dataset class**

Write a `SNIPSDataset` class that inherits from `torch.utils.data.Dataset`.
It must implement three methods:

- `__init__`: takes a list of raw utterances and labels, applies the
  preprocessing pipeline (clean → encode → label-encode), stores results
  as tensors
- `__len__`: returns the number of examples
- `__getitem__`: returns a single (encoded_sentence_tensor, label_tensor) pair

Each encoded sentence will be a `LongTensor` of shape `[max_seq_len]`.
Each label will be a scalar `LongTensor`.

**Step 2 — DataLoader construction**

Create three DataLoaders: one for training, one for validation, one for test.

Key settings:
- `batch_size = 32` (standard starting point; fits easily in CPU memory)
- `shuffle = True` for training (prevents the model from learning order effects)
- `shuffle = False` for validation and test (order doesn't matter, reproducibility does)

**Step 3 — Batch inspection**

Iterate one batch from the training DataLoader and print:
- Shape of the input tensor: should be `[32, max_seq_len]`
- Shape of the label tensor: should be `[32]`
- A decoded version of the first 3 sentences in the batch to confirm they
  look like real SNIPS utterances

### Why this phase matters

PyTorch's DataLoader handles batching, shuffling, and (optionally) parallel
data loading automatically. Wrapping data in a Dataset class also makes it
trivial to swap in a different dataset later without changing any training code.

### Verification checklist before approving Phase 4

- [ ] `len(train_dataset)` returns the expected number of training examples
- [ ] `train_dataset[0]` returns a tuple of two tensors with correct shapes
- [ ] One training batch has shape `[32, max_seq_len]` for inputs and `[32]` for labels
- [ ] Decoded sentences from the batch look like real utterances (not garbage)
- [ ] Label values are integers in the range 0–6

---

## Phase 4 — Model Architecture

**Goal:** Build the LSTM classifier as a clean PyTorch `nn.Module`.

### What will be done

All work in this phase lives in `src/model.py`.

**The model class: `IntentClassifier`**

The constructor accepts these hyperparameters (with sensible defaults):

| Parameter | Default | What it controls |
|---|---|---|
| `vocab_size` | from preprocessing | number of rows in embedding table |
| `embed_dim` | 128 | size of each word vector |
| `hidden_size` | 256 | size of LSTM hidden state |
| `num_layers` | 1 | number of stacked LSTM layers |
| `num_classes` | 7 | number of intent categories |
| `dropout` | 0.3 | fraction of neurons zeroed during training |
| `bidirectional` | False | whether to run LSTM in both directions |
| `pad_idx` | 0 | tells embedding layer to output zeros for PAD tokens |

**The forward pass (step by step):**

```
Input: x  shape [batch_size, seq_len]  (integer indices)

1. Embedding:
   out = self.embedding(x)
   shape → [batch_size, seq_len, embed_dim]

2. LSTM:
   out, (h_n, c_n) = self.lstm(out)
   h_n shape → [num_layers * num_directions, batch_size, hidden_size]

3. Extract sentence representation:
   Take h_n[-1] → the hidden state from the last layer, last timestep
   shape → [batch_size, hidden_size]
   (if bidirectional, concatenate forward and backward: shape → [batch_size, hidden_size*2])

4. Dropout:
   out = self.dropout(out)

5. Linear (Dense):
   logits = self.fc(out)
   shape → [batch_size, num_classes]

6. Return logits (softmax is NOT applied here — CrossEntropyLoss handles it)
```

**A note on bidirectional:**

When `bidirectional=True`, the LSTM produces hidden states in both directions.
The final representation is the concatenation of the forward hidden state
(which has seen the whole sentence left to right) and the backward hidden state
(which has seen the whole sentence right to left). The Linear layer input size
must be `hidden_size * 2` in this case.

**Step 2 — Model summary**

Instantiate the model and print:
- Total number of trainable parameters
- Layer-by-layer breakdown (using a simple loop or `torchinfo`)

This confirms the architecture is wired correctly before training.

**Step 3 — Forward pass smoke test**

Create a dummy batch of random integer tensors with shape `[32, max_seq_len]`
and pass it through the model. Verify the output shape is `[32, 7]`.

### Why this phase matters

Fixing architecture bugs before training saves enormous time. A shape mismatch
caught in a smoke test takes 10 seconds to fix. The same bug caught after
30 minutes of training means starting over.

### Verification checklist before approving Phase 5

- [ ] Model instantiates without errors
- [ ] Forward pass on dummy input produces output of shape `[batch_size, 7]`
- [ ] Total parameter count is printed and looks reasonable (~1–3M params)
- [ ] Changing `bidirectional=True` and re-running the smoke test still works

---

## Phase 5 — Training Loop

**Goal:** Train the model to convergence with proper discipline: loss tracking,
validation monitoring, learning rate scheduling, and checkpointing.

### What will be done

All work in this phase lives in `src/train.py`.

**Hyperparameters (locked in before training starts):**

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | Adam | adaptive learning rate, robust default |
| Learning rate | 1e-3 | standard Adam starting point |
| Loss function | CrossEntropyLoss | standard for multi-class classification |
| Batch size | 32 | set in Phase 3 |
| Max epochs | 30 | early stopping will likely trigger before this |
| Early stopping patience | 5 | stop if val accuracy doesn't improve for 5 epochs |
| LR scheduler | ReduceLROnPlateau | halve LR if val loss plateaus |

**The training loop structure:**

```
for each epoch:

    --- TRAINING ---
    set model to train mode (dropout active)
    for each batch in train_dataloader:
        zero gradients
        forward pass → logits
        compute loss (CrossEntropyLoss)
        backward pass (compute gradients)
        optimizer step (update weights)
        accumulate batch loss and correct predictions

    compute epoch training loss and accuracy

    --- VALIDATION ---
    set model to eval mode (dropout disabled)
    with torch.no_grad():
        for each batch in val_dataloader:
            forward pass → logits
            compute loss
            accumulate batch loss and correct predictions

    compute epoch validation loss and accuracy
    update learning rate scheduler

    --- LOGGING ---
    print epoch summary:
    "Epoch 5/30 | Train Loss: 0.312 | Train Acc: 89.2% | Val Loss: 0.287 | Val Acc: 91.4%"

    --- CHECKPOINTING ---
    if val_accuracy > best_val_accuracy:
        save model weights to checkpoints/best_model.pt
        update best_val_accuracy

    --- EARLY STOPPING ---
    if val_accuracy hasn't improved for `patience` epochs:
        print "Early stopping triggered"
        break
```

**Saving and loading:**

The checkpoint saves the full model state including:
- `model.state_dict()` — all weights
- `epoch` — which epoch achieved this performance
- `val_accuracy` — so we know how good this checkpoint is
- `hyperparameters` — so we can reconstruct the model later

**Plotting training curves:**

After training completes, generate and save two plots:
- Train loss vs Validation loss over epochs
- Train accuracy vs Validation accuracy over epochs

These curves are the primary diagnostic tool. Diverging train/val curves
indicate overfitting. A flat val curve that stops improving before train
accuracy does also signals overfitting.

### Why this phase matters

The training loop is where most silent bugs live. Forgetting `optimizer.zero_grad()`
causes gradient accumulation. Forgetting `model.eval()` during validation
keeps dropout active and gives misleadingly bad validation numbers.
Writing this carefully and verifying each component independently prevents
hours of debugging.

### Verification checklist before approving Phase 6

- [ ] Loss decreases over the first 3–5 epochs (model is learning)
- [ ] Validation accuracy is being tracked separately from training accuracy
- [ ] `checkpoints/best_model.pt` exists after training
- [ ] Training curves are plotted and saved
- [ ] Early stopping triggered correctly (or training completed all epochs)
- [ ] Final best validation accuracy is noted: _______%

---

## Phase 6 — Evaluation

**Goal:** Measure the true generalization performance of the model on the
held-out test set. Generate a full suite of classification metrics and a
confusion matrix.

### What will be done

All work in this phase lives in `src/evaluate.py`.

**Step 1 — Load best checkpoint**

Load the saved weights from `checkpoints/best_model.pt`. Reconstruct the model
with the same hyperparameters used during training. Set to eval mode.

**Step 2 — Collect predictions**

Run the model on the entire test set. Collect:
- All predicted class indices
- All true class indices
- All confidence scores (max softmax probability per example)

**Step 3 — Classification metrics**

Using `sklearn.metrics`, compute:

| Metric | What it measures |
|---|---|
| Accuracy | % of all predictions that are correct |
| Precision (macro) | avg across classes: of all times we predicted class X, how often were we right |
| Recall (macro) | avg across classes: of all actual class X examples, how many did we catch |
| F1 (macro) | harmonic mean of precision and recall, per class then averaged |

Print a full `classification_report` showing per-class breakdown.

**Step 4 — Confusion matrix**

Generate a 7×7 confusion matrix heatmap. Each cell (i, j) shows how many
examples of true intent i were predicted as intent j. The diagonal is correct
predictions; off-diagonal entries reveal specific confusions.

This is the most informative diagnostic. For example, if BookRestaurant and
SearchScreeningEvent are frequently confused, that's an actionable insight
about what the model hasn't learned.

**Step 5 — Error analysis**

Find examples where the model was wrong and confident (high confidence,
wrong prediction). Print 5–10 of these. These are the most interesting
failure cases and reveal systematic problems the model has.

**Step 6 — Expected performance targets**

A well-trained LSTM on SNIPS should achieve:
- Accuracy: 95–98% (SNIPS is a relatively clean, well-separated dataset)
- Macro F1: > 0.95

If we fall below 90%, we re-examine the preprocessing pipeline and
hyperparameters before declaring the model done.

### Verification checklist before approving Phase 7

- [ ] All 4 metrics (accuracy, precision, recall, F1) are printed
- [ ] Per-class breakdown is visible (no class has drastically lower F1)
- [ ] Confusion matrix heatmap is saved as an image
- [ ] At least 5 error examples are printed and inspected
- [ ] Overall test accuracy is noted: _______%

---

## Phase 7 — Inference Interface

**Goal:** Wrap the trained model in a clean, usable predict function and an
interactive chatbot loop for demonstration.

### What will be done

All work in this phase lives in `src/predict.py`.

**Step 1 — `predict(sentence)` function**

Write a function that takes a raw string and returns the intent and confidence:

```python
def predict(sentence: str) -> dict:
    # 1. Load word2idx, idx2label, max_seq_len from saved artifacts
    # 2. Clean and tokenize the sentence
    # 3. Encode to integer sequence with padding
    # 4. Convert to tensor, add batch dimension
    # 5. Forward pass through model (eval mode, no_grad)
    # 6. Apply softmax to get probabilities
    # 7. Return predicted intent and confidence

    return {
        "input": sentence,
        "intent": "PlayMusic",
        "confidence": 0.974,
        "all_scores": {"PlayMusic": 0.974, "GetWeather": 0.012, ...}
    }
```

**Step 2 — Interactive chatbot loop**

```
=== Intent Classification Chatbot ===
Type a sentence to classify. Type 'quit' to exit.

You: play some jazz music
Intent:     PlayMusic
Confidence: 97.4%

You: what is the weather in bangalore tomorrow
Intent:     GetWeather
Confidence: 95.8%

You: quit
Goodbye!
```

**Step 3 — Batch prediction test**

Run 20 manually written test sentences (5 from each of 4 different intents,
none from the training set) through `predict()` and print the results.
This is a final sanity check that the model generalizes to completely new phrasing.

### Verification checklist before approving Phase 8

- [ ] `predict("play some jazz")` returns a dict with correct keys
- [ ] Confidence scores sum to 1.0 across all 7 classes
- [ ] Interactive loop runs without crashing
- [ ] At least 15 out of 20 batch test sentences are correctly classified

---

## Phase 8 — Reflection and Documentation

**Goal:** Document what was built, what the results show, and what the
limitations reveal about why transformers were invented.

### What will be done

**Step 1 — Update README.md** with:
- Project overview
- How to set up the environment
- How to run training
- How to run the chatbot
- Final test accuracy achieved

**Step 2 — Limitations analysis**

After evaluation, explicitly document observed failure modes:

- Does accuracy drop on longer, more complex sentences?
- Are there specific word pairs that confuse the model (e.g., "book" in
  different contexts)?
- Is the model sensitive to word order, or does it sometimes ignore it?

**Step 3 — Bridge to transformers**

Write a short note (in the README or a separate `REFLECTIONS.md`) connecting
the LSTM's limitations to transformer design choices:

| LSTM limitation | Transformer solution |
|---|---|
| Processes tokens sequentially, slow to train | Processes all tokens in parallel |
| Long-range dependencies can fade | Attention directly connects any two tokens |
| Final hidden state is an information bottleneck | Attention over all positions, not just the last |
| No pre-trained language understanding | Pre-trained on massive corpora (BERT, GPT) |

### Verification checklist before project completion

- [ ] README is complete and another person could run the project from it
- [ ] At least 3 specific failure cases are documented
- [ ] The LSTM-to-transformer bridge section is written
- [ ] All source files are clean and commented

---

## Summary Table

| Phase | File(s) | Key output | Approval needed |
|---|---|---|---|
| 0 — Environment | `requirements.txt` | Working env + raw data | ✅ Yes |
| 1 — Exploration | `notebooks/01_exploration.ipynb` | Agreed `max_seq_len` & `vocab_size` | ✅ Yes |
| 2 — Preprocessing | `src/preprocess.py` | `word2idx`, `idx2label`, encoder functions | ✅ Yes |
| 3 — Dataset | `src/dataset.py` | DataLoaders with correct tensor shapes | ✅ Yes |
| 4 — Model | `src/model.py` | Model passing forward pass smoke test | ✅ Yes |
| 5 — Training | `src/train.py` | `best_model.pt` + training curves | ✅ Yes |
| 6 — Evaluation | `src/evaluate.py` | Accuracy, F1, confusion matrix | ✅ Yes |
| 7 — Inference | `src/predict.py` | Working chatbot loop | ✅ Yes |
| 8 — Reflection | `README.md` | Documented limitations + transformer bridge | ✅ Yes |

---

*Plan version 1.0 — awaiting Phase 0 approval to begin.*