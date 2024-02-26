#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <dlfcn.h>
#include <vector>

#include "../../faster_rwkvd.h"

#define CHECK()                                             \
{                                                           \
    const char* dlsym_error = dlerror();                    \
    if (dlsym_error) {                                      \
        std::cerr << "Cannot load libray or symbol: "            \
             << dlsym_error << '\n';                        \
        exit(1);                                            \
    }                                                       \
}

static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;
const int pad_token_id = 0;
const int bos_token_id = 2;
const int eos_token_id = 3;

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " model_path strategy input_file" << std::endl;
    return 1;
  } else {
    std::cout << "model_path: " << argv[1] << std::endl;
    std::cout << "strategy: " << argv[2] << std::endl;
    std::cout << "input_file: " << argv[3] << std::endl;
  }
  std::cout.setf(std::ios::unitbuf);
  std::cout << "Loading libfaster_rwkvd.so" << std::endl;
  void *handle = dlopen("libfaster_rwkvd.so", RTLD_LAZY);
  CHECK();

  rwkv_model_t (*model_create)(const char*, const char*);
  rwkv_tokenizer_t (*tokenizer_create)();
  rwkv_sampler_t (*sampler_create)();
  char (*model_run)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char, float, int, float);
  model_create = (rwkv_model_t (*)(const char*, const char*))dlsym(handle, "rwkv_model_create");
  CHECK();
  tokenizer_create = (rwkv_tokenizer_t (*)())dlsym(handle, "rwkv_ABCTokenizer_create");
  CHECK();
  sampler_create = (rwkv_sampler_t (*)())dlsym(handle, "rwkv_sampler_create");
  CHECK();
  model_run = (char (*)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char, float, int, float))dlsym(handle, "rwkv_abcmodel_run_with_tokenizer_and_sampler");
  CHECK();

  rwkv_model_t model = model_create(argv[1], argv[2]);
  rwkv_tokenizer_t tokenizer = tokenizer_create();
  rwkv_sampler_t sampler = sampler_create();

  std::ifstream ifs(argv[3]);
  std::stringstream buffer;

  buffer << ifs.rdbuf();
  std::string input = buffer.str();
  input.erase(input.find_last_not_of(" \t\n\r\f\v") + 1);
  std::cout << input;
  static const int N_TRIAL = 1;
  for (int t = 0; t < N_TRIAL; t++) {
    std::string result = input;
    auto start = std::chrono::system_clock::now();

    char output = model_run(model, tokenizer, sampler, bos_token_id, 1.f, 1, 0.f);
    for (auto i : input) {
      output = model_run(model, tokenizer, sampler, i, 1.f, 1, 0.f);
    }

    for (int i = 0; i < 1024; i++) {
      std::cout << output;
      result += output;
      output = model_run(model, tokenizer, sampler, output, 1.f, 1, 0.f);
    }
    std::cout << std::endl;
    auto end = std::chrono::system_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);
    if (kShowSpeed) {
      std::cout << "time: " << total_time.count() << "ms" << std::endl;
      std::cout << "num tokens: " << result.size() << std::endl;
      std::cout << "ms per token: " << 1. * total_time.count() / result.size() << std::endl;
      std::cout << "tokens per second: " << 1. * result.size() / total_time.count() * 1000 << std::endl;
    }
    std::ofstream ofs("output_" + std::to_string(t) + ".txt");
    ofs << result;
  }

  return 0;
}
