Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

# Project Instructions & Coding Standards
### Intent Classification Chatbot — LSTM on SNIPS

---

> This document is the single source of truth for how this project is structured
> and how code must be written. Every file created, every script written, and every
> experiment run must follow these rules without exception.

---

# Project Instructions
### Intent Classification Chatbot — LSTM on SNIPS

---

> For coding behavior and style rules, refer to the existing behavioral guidelines doc.
> This document covers project structure, file placement, and naming only.

---

## Directory Structure

Create this before writing any code. Every file must live in its designated directory.

```
intent-classifier/
│
├── data/
│   ├── raw/                     ← original SNIPS JSON files, never modified
│   │   ├── train/
│   │   └── test/
│   └── processed/               ← vocab files, label maps, encoded tensors
│
├── src/                         ← finalized, verified source code only
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── notebooks/                   ← exploration and results analysis only
│   ├── 01_exploration.ipynb
│   └── 02_results_analysis.ipynb
│
├── experiments/                 ← one subfolder per training run
│   └── exp_001_baseline/
│       ├── config.json          ← every hyperparameter used in this run
│       ├── training_log.csv     ← epoch-by-epoch loss and accuracy
│       └── plots/
│
├── models/                      ← .pt checkpoint files only
│   ├── best_model.pt
│   └── README.md                ← which experiment produced this + its accuracy
│
├── temp/                        ← scratch scripts, throwaway experiments
│   └── .gitkeep
│
├── assets/                      ← saved plots referenced in README
│
├── PLANS.md
├── INSTRUCTIONS.md
├── requirements.txt
└── README.md
```

---

## What Each Directory Is For

**`data/raw/`** — Read-only after download. Never modify, clean, or overwrite anything here.

**`data/processed/`** — Outputs of `preprocess.py`: `word2idx.json`, `idx2label.json`, and any encoded data files.

**`src/`** — Code that has been tested and verified against a phase checklist. A file moves here from `temp/` only once it's confirmed working. Notebooks may import from `src/`, never the reverse.

**`notebooks/`** — EDA and visualization only. No model definitions, no training loops, no preprocessing logic. All of that lives in `src/` and gets imported.

**`experiments/`** — Every distinct hyperparameter configuration gets its own numbered folder. Never overwrite a previous experiment. This is how you compare runs later.

**`models/`** — Checkpoints only. Update `models/README.md` every time `best_model.pt` is replaced, noting which experiment it came from and what accuracy it achieved.

**`temp/`** — Anything goes. One hard rule: nothing in `temp/` is ever imported by `src/` or `notebooks/`. Add `temp/*` to `.gitignore` (keep `temp/.gitkeep`).

**`assets/`** — Any plot that needs to appear in the README gets saved here so it persists outside of notebooks.

---

## File Naming

| What | Convention | Example |
|---|---|---|
| Python source files | lowercase, underscores | `preprocess.py` |
| Notebooks | numbered + descriptive | `01_exploration.ipynb` |
| Experiment folders | zero-padded + description | `exp_001_baseline` |
| Checkpoints | descriptive + metric | `best_model_val97.pt` |
| Experiment configs | always this name | `config.json` |
| Plots | lowercase, underscores | `confusion_matrix.png` |

No spaces. No `final`, `final_v2`, `copy`, `new` in any filename.

---

## Experiment Config

Every training run must save a `config.json` before training starts:

```json
{
  "seed": 42,
  "embed_dim": 128,
  "hidden_size": 256,
  "num_layers": 1,
  "bidirectional": false,
  "dropout": 0.3,
  "batch_size": 32,
  "learning_rate": 0.001,
  "max_epochs": 30,
  "early_stopping_patience": 5,
  "max_seq_len": 16,
  "vocab_size": 4000
}
```

If any value changes, it's a new experiment with a new folder — not an overwrite.

---

## Git Commits

One commit per verified phase. Message format:

```
[Phase 0] Add requirements.txt and project structure
[Phase 2] Add vocab builder and sentence encoder
[Phase 4] Fix bidirectional hidden state extraction
```

No commits mid-phase. A commit means the phase checklist passed.

---

## Before Writing Any File

- [ ] Does the target directory exist?
- [ ] Is this the right directory for this file?
- [ ] If training: is a new `experiments/exp_NNN_*/` folder created?
- [ ] If a new checkpoint: is `models/README.md` updated?
- [ ] If graduating from `temp/` to `src/`: has it passed the phase checklist?