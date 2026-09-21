import numpy as np

from utils.sigmoid import sigmoid

def forward_pass(center, context, negatives, w_in, w_out):
    v_c = w_in[center]
    u_o = w_out[context]
    u_ni = w_out[negatives]

    pos_score = sigmoid(np.dot(u_o, v_c))
    neg_score = sigmoid(np.dot(u_ni, v_c))

    # + 1e-10 is a safety to avoid log(0)
    loss = -np.log(pos_score + 1e-10) - np.sum(np.log(1-neg_score + 1e-10))
    return pos_score, neg_score, loss


def gradient(center, context, negatives, pos_score, neg_score, w_in, w_out):
    v_c = w_in[center]
    u_o = w_out[context]
    u_ni = w_out[negatives]

    grad_u_o = (pos_score - 1) * v_c
    grad_u_ni = neg_score[:, None] * v_c[None, :]
    # FIX: added axis=0 -- without it, np.sum collapses (K,D) into a single
    # scalar instead of summing across the K negatives into a (D,) vector.
    grad_v_c = (pos_score - 1) * u_o + np.sum(neg_score[:, None] * u_ni, axis=0)

    return grad_u_o, grad_u_ni, grad_v_c


def update_params(center, context, negatives, grad_v_c, grad_u_o, grad_u_ni, w_in, w_out, lr):
    w_in[center]   -= lr * grad_v_c
    w_out[context] -= lr * grad_u_o
    np.add.at(w_out, negatives, -lr * grad_u_ni)
    return w_in, w_out


def current_lr(epoch, epochs, lr):
    return lr * max(1e-4, 1 - epoch/epochs)
