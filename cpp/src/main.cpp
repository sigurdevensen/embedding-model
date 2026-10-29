#include <iostream>
#include <vector>
#include <string>
#include <sstream>

std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::stringstream ss(text);
    std::string sentence;

    while (std::getline(ss, sentence, '.')) {
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }

    return sentences;
}

int main() {
    std::vector<std::string> sentences = split_sentences("This is the first sentence. This is the second sentence. And this is the third sentence.");
    for (const auto& sentence : sentences) {
        std::cout << sentence << std::endl;
    }
    return 0;
}
