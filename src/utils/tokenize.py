import re
import string
import numpy as np


def split_sentences(text: str) -> list:
    """Split raw text into sentences BEFORE punctuation is stripped —
    tokenize() removes periods, so this has to run first."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def tokenize(text: str) -> list:
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = clean_text.lower().split()
    return tokens

def load_gutenberg(path: str) -> str:
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig strips the BOM
        raw = f.read()
    start = re.search(r'\*{3} START OF (THIS|THE) PROJECT GUTENBERG EBOOK', raw)
    end = re.search(r'\*{3} END OF (THIS|THE) PROJECT GUTENBERG EBOOK', raw)
    if start:
        raw = raw[raw.index("\n", start.start()) + 1:]
        # recalculate end position after slicing
        end = re.search(r'\*{3} END OF (THIS|THE) PROJECT GUTENBERG EBOOK', raw)
    if end:
        raw = raw[:end.start()]
    return raw

def frequency(tokens: list):
    word2idx = {}
    idx2word = {}
    freqs = []

    for word in tokens:
        if word not in word2idx:
            pos = len(freqs)
            word2idx[word] = pos
            idx2word[pos] = word
            freqs.append(1)
        else:
            pos = word2idx[word]
            freqs[pos] += 1

    return word2idx, idx2word, freqs

def generate_pairs(id_sentences, keep_prob, max_window):
    rng = np.random.default_rng()
    pairs = []

    for sentence in id_sentences:
        surviving = [x for x in sentence if rng.random() < keep_prob[x]]

        for i, center in enumerate(surviving):
            m = rng.integers(1, max_window+1)
            start = max(0, i-m)
            end = min(len(surviving), i+m+1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, surviving[j]))

    return pairs