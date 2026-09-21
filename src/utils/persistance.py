import numpy as np
import os

def save_vocab(word2idx, idx2word, prefix="vocab"):
    os.makedirs("out", exist_ok=True)
    np.save(f"out/{prefix}_word2idx.npy", word2idx)
    np.save(f"out/{prefix}_idx2word.npy", idx2word)


def load_vocab(prefix="vocab"):
    word2idx = np.load(f"out/{prefix}_word2idx.npy", allow_pickle=True).item()
    idx2word = np.load(f"out/{prefix}_idx2word.npy", allow_pickle=True).item()
    return word2idx, idx2word