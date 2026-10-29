import numpy as np

def subsample_probs(freqs, threshold=1e-2):  # threshold = 1e-3
    freqs = np.array(freqs)
    f = freqs / freqs.sum()
    keep_prob = (np.sqrt(f / threshold) + 1) * (threshold / f)

    return np.clip(keep_prob, 0, 1)


def negative_sampling(freqs: list) -> list:
    freqs = np.array(freqs)
    powered = freqs**(3/4)

    return powered/powered.sum()