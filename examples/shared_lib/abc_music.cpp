#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <vector>

#include "../../faster_rwkvd.h"

#define CHECK()                                             \
{                                                           \
    const char* dlsym_error = dlerror();                    \
    fprintf(stderr,                                     \
          "Cannot load libray or symbol at %s:%d: %s\n", \
          __FILE__,__LINE__,                             \
          dlsym_error?:"Unknown Error");                 \
    exit(1);                                            \
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

#ifdef _WIN32
  std::cout << "Loading faster_rwkvd.dll" << std::endl;
  HMODULE handle = LoadLibrary("faster_rwkvd.dll");
  if (handle == NULL) {
    std::cerr << "Cannot load library: " << GetLastError() << std::endl;
    return 1;
  }
#else
  std::cout << "Loading libfaster_rwkvd.so" << std::endl;
  void *handle = dlopen("libfaster_rwkvd.so", RTLD_NOW);
  if(!handle)
    CHECK();
#endif

  rwkv_model_t (*model_create)(const char*, const char*);
  rwkv_tokenizer_t (*tokenizer_create)();
  rwkv_sampler_t (*sampler_create)();
  char (*model_run)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char, float, int, float);
  char (*model_run_prompt)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char*, const int, float, int, float);
  void (*clear_states)(rwkv_model_t);

#ifdef _WIN32
  model_create = (rwkv_model_t (*)(const char*, const char*))GetProcAddress(handle, "rwkv_model_create");
  tokenizer_create = (rwkv_tokenizer_t (*)())GetProcAddress(handle, "rwkv_ABCTokenizer_create");
  sampler_create = (rwkv_sampler_t (*)())GetProcAddress(handle, "rwkv_sampler_create");
  model_run = (char (*)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char, float, int, float))GetProcAddress(
      handle, "rwkv_abcmodel_run_with_tokenizer_and_sampler");
  model_run_prompt = (char (*)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char*, const int, float, int, float))GetProcAddress(
      handle, "rwkv_abcmodel_run_prompt");
  clear_states = (void (*)(rwkv_model_t))GetProcAddress(handle, "rwkv_model_clear_states");
  if (model_create == NULL || tokenizer_create == NULL || sampler_create == NULL || model_run == NULL) {
    std::cerr << "Cannot load symbol: " << GetLastError() << std::endl;
    return 1;
  }
#else
  model_create = (rwkv_model_t (*)(const char*, const char*))dlsym(handle, "rwkv_model_create");
  if(!model_create)
    CHECK();
  tokenizer_create = (rwkv_tokenizer_t (*)())dlsym(handle, "rwkv_ABCTokenizer_create");
  if(!tokenizer_create)
    CHECK();
  sampler_create = (rwkv_sampler_t (*)())dlsym(handle, "rwkv_sampler_create");
  if(!sampler_create)
    CHECK();
  model_run = (char (*)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char, float, int, float))dlsym(handle, "rwkv_abcmodel_run_with_tokenizer_and_sampler");
  if(!model_run)
    CHECK();
  model_run_prompt = (char (*)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char*, const int, float, int, float))dlsym(handle, "rwkv_abcmodel_run_prompt");
  if(!model_run_prompt)
    CHECK();
  clear_states = (void (*)(rwkv_model_t))dlsym(handle, "rwkv_model_clear_states");
  if(!clear_states)
    CHECK();
#endif

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
    // IMPORTANT: clear states before each round of generation
    clear_states(model);
    std::string result = input;
    auto start = std::chrono::system_clock::now();

    char output = model_run_prompt(model, tokenizer, sampler, input.c_str(), input.length(), 1.f, 1, 0.f);

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
