# Embedding Model

An implementation of Word2Vec (Skip-Gram with Negative Sampling), using only NumPy. Every gradient is derived by hand.
The code is written manually, however shoutout to Claude for helping me with the math.

## Requirements

- Python 3.9+
- NumPy

```bash
pip install numpy
```

## Usage

```bash
python main.py
```

## How it works

The pipeline runs in three stages:

**1. Data prep** - `split_sentences` -> `tokenize` -> `frequency` turn raw text into a vocabulary (`word2idx`/`idx2word`) and per-word counts, keeping sentence boundaries intact so training pairs never cross between sentences.

**2. Sampling** - `subsample_probs` downweights frequent words (like "the") before pairs are built; `negative_sampling` builds the noise distribution (`count(w)^0.75`) used to draw negative examples during training. `generate_pairs` then produces `(center, context)` pairs per sentence, using a random window size per center word.

**3. Model + training** - two embedding matrices (`W_in`, `W_out`) are trained with plain SGD: `forward_pass` scores a pair with sigmoid, `gradient` computes the hand-derived gradients, `update_params` applies them. `current_lr` linearly decays the learning rate over training. `train` wires all of this into the full loop, run once per epoch, once per training pair.

**Output**: `W_in`, the trained embedding matrix - one row per vocabulary word. `most_similar` looks up a word's nearest neighbors by cosine similarity.