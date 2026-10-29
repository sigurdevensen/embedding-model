#include "utils/tokenize.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string text = tokenize("Hello world! This is a test. Let's see how it works.");
    std::cout << text << std::endl;

    std::vector<std::string> sentences = split_sentences(text);

    for (const auto& sentence : sentences) {
        std::cout << sentence << std::endl;
    }

    Tokens tokens = frequency(text);
    std::cout << "Vocab size: " << tokens.word2idx.size() << " | Total tokens " << tokens.freqs.size() << " | Sentences " << sentences.size() << std::endl;
    return 0;
}
