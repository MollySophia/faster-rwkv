#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
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

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " tokenizer_path model_path strategy input_file" << std::endl;
    return 1;
  } else {
    std::cout << "tokenizer_path: " << argv[1] << std::endl;
    std::cout << "model_path: " << argv[2] << std::endl;
    std::cout << "strategy: " << argv[3] << std::endl;
    std::cout << "input_file: " << argv[4] << std::endl;
  }
  std::cout.setf(std::ios::unitbuf);

#ifdef _WIN32
  std::cout << "Loading faster_rwkvd.dll" << std::endl;
  HMODULE handle = LoadLibrary("lib\\fastmodel\\faster_rwkvd.dll");
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
  char (*model_run_prompt_file)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char*, const int, const char*, const int, const int, float, int, float);

#ifdef _WIN32
  model_create = (rwkv_model_t (*)(const char*, const char*))GetProcAddress(handle, "rwkv_model_create");
  tokenizer_create = (rwkv_tokenizer_t (*)())GetProcAddress(handle, "rwkv_ABCTokenizer_create");
  sampler_create = (rwkv_sampler_t (*)())GetProcAddress(handle, "rwkv_sampler_create");
  model_run_prompt_file = (char (*)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char*, const int, const char*, const int, const int, float, int, float))GetProcAddress(
      handle, "rwkv_midimodel_run_prompt_from_file");
  if (model_create == NULL || tokenizer_create == NULL || sampler_create == NULL || model_run_prompt_file == NULL) {
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
  model_run_prompt_file = (char (*)(rwkv_model_t, rwkv_tokenizer_t, rwkv_sampler_t, const char*, const int, const char*, const int, const int, float, int, float))dlsym(handle, "rwkv_midimodel_run_prompt_from_file");
  if(!model_run_prompt_file)
    CHECK();
#endif

  rwkv_model_t model = model_create(argv[1], argv[2]);
  rwkv_tokenizer_t tokenizer = tokenizer_create();
  rwkv_sampler_t sampler = sampler_create();

  model_run_prompt_file(model, tokenizer, sampler, argv[4], strlen(argv[4]), "output.mid", strlen("output.mid"), 1024, 1.0, 0, 0.0);

  return 0;
}
