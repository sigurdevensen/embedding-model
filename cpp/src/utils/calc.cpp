#include <vector>
#include "tokenize.hpp"

struct Pass_scores {
    std::vector<int> pos_score;
    std::vector<int> new_score;
    int loss;
};

struct Gradients {
    std::vector<double> grad_u_o;
    std::vector<double> grad_u_ni;
    std::vector<double> grad_v_c;
};

Pass_scores forward_pass(int center, int context, int negatives, std::vector<int> &w_in, std::vector<int> &w_out) {
    int v_c = w_in[center];
    int u_o = w_out[context];
    int u_ni = w_out[negatives];

    int pos_score = sigmoid(v_c * u_o);
    int new_score = sigmoid(v_c * u_ni);
}