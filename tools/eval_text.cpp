#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <kernels/kernels.h>
#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " [vocab] [model] [strategy] [text]\n";
        return 1;
    }

    rwkv::Tokenizer tokenizer(argv[1]);
    rwkv::Sampler sampler;
    rwkv::Model model(argv[2], argv[3]);
    char *eval_text_buf;
    std::ifstream eval_text_file(argv[4], std::ios::binary | std::ios::ate);
    size_t file_size;
    if (eval_text_file.is_open()) {
        eval_text_file.seekg(0, std::ios::end);
        file_size = eval_text_file.tellg();
        eval_text_buf = new char[file_size];
        eval_text_file.seekg(0, std::ios::beg);
        eval_text_file.read(eval_text_buf, file_size);
        eval_text_file.close();
    } else {
        std::cerr << "Unable to open file\n";
        return 1;
    }
    std::vector<std::string> eval_text;
    size_t next = 0;
    for (size_t i = 0; i < file_size; i++) {
        if (eval_text_buf[i] == '|') {
            eval_text.push_back(std::string(eval_text_buf + next, i - next));
            next = i + 1;
        }
    }
    delete[] eval_text_buf;
    std::cout << "Eval texts num: " << eval_text.size() << std::endl;

    float xsum = 0;
    int xcnt = 0;
    int xacc = 0;

    for (const auto &text : eval_text) {
        std::cout << "Sample num: " << xcnt << std::endl;
        auto prompt_ids = tokenizer.encode(text.substr(0, text.find_last_of(' ')));
        auto target_ids = tokenizer.encode(text.substr(text.find_last_of(' ')));
        std::cout << "Prompt: " << text.substr(0, text.find_last_of(' ')) << std::endl;
        std::cout << "Target: " << text.substr(text.find_last_of(' ')) << std::endl;
        model.ResetStates();
        std::cout << "Response: ";

        bool correct = true;
        float logits = 0;
        auto output = Copy(model.Run(prompt_ids), rwkv::Device::kCPU);
        auto probs = softmax(output, 1.f);
        for (int i = 0; i < target_ids.size(); i++) {
            auto output_id = sampler.Sample(output, /*temperature=*/1.f, /*top_k=*/1, 1);
            logits += std::log(probs.data_ptr<float>()[target_ids[i]]);
            if (output_id != target_ids[i]) {
                correct = false;
            }
            std::cout << tokenizer.decode(output_id);

            output = Copy(model.Run(target_ids[i]), rwkv::Device::kCPU);
            probs = softmax(output, 1.f);
        }

        xcnt++;
        if (correct) {
            xacc++;
        } 
        xsum += logits;

        // if (xcnt % 10 == 0) {
            std::cout << "\nAccuracy: " << xacc << "/" << xcnt << " = " << (float)xacc / xcnt << std::endl;
            std::cout << "Perplexity: " << std::exp(-xsum / xcnt) << std::endl;
            std::cout << "====================================\n";
        // }
    }

    
    return 0;
}
