import os
from multiprocessing import Pool, shared_memory

import numpy as np

from utils.tokenize import load_gutenberg, split_sentences, tokenize, frequency, generate_pairs
from utils.sampling import subsample_probs, negative_sampling
from utils.persistance import save_vocab
from utils.calc import forward_pass, gradient, update_params, current_lr


def init_embeddings(vocab_size, dim, seed=42):
    rng = np.random.default_rng(seed)
    W_in = (rng.random((vocab_size, dim)) - 0.5) / dim   # small random values
    W_out = np.zeros((vocab_size, dim))                  # zeros is fine here
    return W_in, W_out

def _train_chunk_mp(args):
    """Hogwild worker: attaches to shared memory and updates w_in/w_out in-place."""
    pairs, neg_table, k, lr, shm_in_name, shm_out_name, shape_in, shape_out, dtype = args
    shm_in  = shared_memory.SharedMemory(name=shm_in_name)
    shm_out = shared_memory.SharedMemory(name=shm_out_name)
    w_in  = np.ndarray(shape_in,  dtype=dtype, buffer=shm_in.buf)
    w_out = np.ndarray(shape_out, dtype=dtype, buffer=shm_out.buf)
    rng = np.random.default_rng()
    n = len(neg_table)
    total_loss = 0.0
    for center, context in pairs:
        negatives = neg_table[rng.integers(0, n, size=k)]
        pos_score, neg_score, loss = forward_pass(center, context, negatives, w_in, w_out)
        grad_u_o, grad_u_ni, grad_v_c = gradient(center, context, negatives, pos_score, neg_score, w_in, w_out)
        update_params(center, context, negatives, grad_v_c, grad_u_o, grad_u_ni, w_in, w_out, lr)
        total_loss += loss
    shm_in.close()
    shm_out.close()
    return total_loss


def train(id_sentences, word2idx, freqs, dim=768, epochs=200, k=5, lr=0.05, max_window=4, seed=42, num_workers=None):
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)

    rng = np.random.default_rng(seed)
    vocab_size = len(word2idx)

    keep_prob = subsample_probs(freqs)
    neg_probs = negative_sampling(freqs)
    neg_table = rng.choice(vocab_size, size=1_000_000, p=neg_probs)

    w_in, w_out = init_embeddings(vocab_size, dim, seed=seed)

    # Put weight matrices in shared memory so worker processes can update them in-place (Hogwild).
    shm_in  = shared_memory.SharedMemory(create=True, size=w_in.nbytes)
    shm_out = shared_memory.SharedMemory(create=True, size=w_out.nbytes)
    shared_w_in  = np.ndarray(w_in.shape,  dtype=w_in.dtype,  buffer=shm_in.buf)
    shared_w_out = np.ndarray(w_out.shape, dtype=w_out.dtype, buffer=shm_out.buf)
    shared_w_in[:]  = w_in
    shared_w_out[:] = w_out

    print(f"Training with {num_workers} workers ...")
    try:
        with Pool(num_workers) as pool:
            for epoch in range(epochs):
                pairs  = generate_pairs(id_sentences, keep_prob, max_window)
                rng.shuffle(pairs)
                cur_lr = current_lr(epoch, epochs, lr)

                chunk_size = max(1, len(pairs) // num_workers)
                chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

                args = [
                    (chunk, neg_table, k, cur_lr,
                     shm_in.name, shm_out.name,
                     shared_w_in.shape, shared_w_out.shape, shared_w_in.dtype)
                    for chunk in chunks
                ]
                total_loss = sum(pool.map(_train_chunk_mp, args))

                if pairs:
                    print(f"epoch {epoch:4d}  avg loss {total_loss/len(pairs):.4f}  lr {cur_lr:.4f}  #pairs {len(pairs)}")
    finally:
        result_w_in  = shared_w_in.copy()
        result_w_out = shared_w_out.copy()
        shm_in.close();  shm_in.unlink()
        shm_out.close(); shm_out.unlink()

    return result_w_in, result_w_out

def most_similar(word, word2idx, idx2word, w_in, topn=5):
    Wn = w_in / np.linalg.norm(w_in, axis=1, keepdims=True)   # normalize every row to unit length
    v = Wn[word2idx[word]]
    sims = Wn @ v                                              # cosine similarity to every word at once
    order = np.argsort(-sims)
    return [(idx2word[i], sims[i]) for i in order if idx2word[i] != word][:topn]


def main():
    corpus_path = "data/2600-0.txt"
    print(f"Loading corpus from {corpus_path} ...")
    text = load_gutenberg(corpus_path)

    print("Splitting into sentences ...")
    sentences = [tokenize(s) for s in split_sentences(text)]
    sentences = [s for s in sentences if s]  # drop empty

    flat_tokens = [tok for sent in sentences for tok in sent]
    word2idx, idx2word, freqs = frequency(flat_tokens)
    print(f"Vocab size: {len(word2idx)}  |  Total tokens: {len(flat_tokens)}  |  Sentences: {len(sentences)}")

    save_vocab(word2idx, idx2word)
    id_sentences = [[word2idx[w] for w in sent] for sent in sentences]

    w_in, w_out = train(id_sentences, word2idx, freqs, dim=100, epochs=5, lr=0.025)
    np.save("w_in.npy", w_in)
    np.save("w_out.npy", w_out)
    print("Saved w_in.npy, w_out.npy, vocab_word2idx.npy, vocab_idx2word.npy")

    for probe in ("cook", "grandmother"):
        if probe in word2idx:
            print(f"Most similar to '{probe}':", most_similar(probe, word2idx, idx2word, w_in))

if __name__ == "__main__":
    main()