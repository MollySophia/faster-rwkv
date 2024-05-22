#include "faster_rwkvd.h"
#include "stdlib.h"
#include "tokenizer.h"
#include "web_rwkv_ffi.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstring>
#include <time.h>

std::string result;
std::string last_out;
std::vector<int> token_ids;
std::map<int, float> occurences;

#ifdef __cplusplus
extern "C" {
#endif

rwkv_model_t rwkv_model_create(const char *path, const char *strategy) {
  init(time(NULL));
  load(path, 0, 0);
  return nullptr;
}

rwkv_tokenizer_t rwkv_ABCTokenizer_create() {
  return new rwkv::ABCTokenizer();
}

rwkv_tokenizer_t rwkv_Tokenizer_create(const char *path) {
  return new rwkv::Tokenizer(path);
}

rwkv_sampler_t rwkv_sampler_create() {
  return nullptr;
}

void rwkv_sampler_set_seed(rwkv_sampler_t sampler_handle, int _seed) {
  seed((uint64_t)_seed);
}

char rwkv_abcmodel_run_with_tokenizer_and_sampler(
    rwkv_model_t model_handle, rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle, const char input,
    // sampler params
    float temperature, int top_k, float top_p) {
  rwkv::ABCTokenizer *tokenizer =
      static_cast<rwkv::ABCTokenizer *>(tokenizer_handle);
  std::string input_str = std::string(1, input);
  auto input_ids = tokenizer->encode(input_str);
  std::vector<uint16_t> input_ids_u16 = std::vector<uint16_t>(input_ids.begin(), input_ids.end());
  uint16_t output_id = infer(input_ids_u16.data(), input_ids_u16.size(), {temperature, top_p, static_cast<uintptr_t>(top_k)});
  if (output_id == tokenizer->eos_token_id) {
    return tokenizer->eos_token_id;
  }
  auto output_str = tokenizer->decode(output_id);
  return output_str[0];
}

char rwkv_abcmodel_run_prompt(
    rwkv_model_t model_handle,
    rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle, const char *input,
    const int input_length,
    // sampler params
    float temperature, int top_k, float top_p) {
  rwkv::ABCTokenizer *tokenizer =
      static_cast<rwkv::ABCTokenizer *>(tokenizer_handle);
  auto input_ids = tokenizer->encode(std::string(input, input_length));
  std::vector<uint16_t> input_ids_u16 = std::vector<uint16_t>(input_ids.begin(), input_ids.end());
  uint16_t output_id = infer(input_ids_u16.data(), input_ids_u16.size(), {temperature, top_p, static_cast<uintptr_t>(top_k)});
  auto output_str = tokenizer->decode(output_id);
  return output_str[0];
}

void rwkv_model_clear_states(rwkv_model_t model_handle) {
  clear_state();
}

#ifdef __cplusplus
}
#endif
