#include "tokenize.hpp"

#include <sstream>
#include <cctype>
#include <algorithm>


std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::stringstream ss(text);
    std::string sentence;

    char ch;
    while (ss.get(ch)) {
        if (ch == '.' || ch == '?' || ch == '!') {
            if (!sentence.empty()) {
                sentences.push_back(sentence);
                sentence.clear();
            }
        } else {
            sentence += ch;
        }
    }

    if (!sentence.empty()) {
        sentences.push_back(sentence);
    }

    return sentences;
}

std::string tokenize(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
    [](unsigned char c) { return std::tolower(c); });

    return text;
}

Tokens frequency(std::string tokens) {
    std::unordered_map<std::string, int> word2idx;
    std::unordered_map<int, std::string> idx2word;
    std::vector<int> freqs;
    std::stringstream token_stream(tokens);
    std::string token;

    while (token_stream >> token) {
        if (word2idx.find(token) == word2idx.end()) {
            int index = word2idx.size();
            word2idx[token] = index;
            idx2word[index] = token;
            freqs.push_back(1);
        } else {
            int index = word2idx[token];
            freqs[index]++;
        }
    }

    return {word2idx, idx2word, freqs};
}
