#include <vector>
#include <random>
#include <cmath>
#include <numeric>

std::vector<double> negative_sampling(const std::vector<int>& freqs) {
    std::vector<double> samples;

    for (std::size_t i = 0; i < freqs.size(); ++i) {
        samples.push_back(std::pow(freqs[i], 3.0 / 4.0));
    }

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        samples[i] /= sum;
    }

    return samples;
}