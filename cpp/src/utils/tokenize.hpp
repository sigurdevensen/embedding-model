#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct Tokens {
    std::unordered_map<std::string, int> word2idx;
    std::unordered_map<int, std::string> idx2word;
    std::vector<int> freqs;
};

std::vector<std::string> split_sentences(const std::string& text);
std::string tokenize(std::string text);
Tokens frequency(std::string tokens);
