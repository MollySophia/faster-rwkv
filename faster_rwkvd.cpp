#include "faster_rwkvd.h"
#include "model.h"
#include "sampler.h"
#include "stdlib.h"
#include "tensor.h"
#include "tokenizer.h"
#include <cstring>
#include <fstream>

#ifdef FR_ENABLE_WEBRWKV
#include <time.h>
#include "web_rwkv_ffi.h"
static bool is_webrwkv = false;
#endif

#ifdef FR_ENABLE_QNN
#include "kernels/qnn/include/librwkv-qualcomm.h"
#include "kernels/qnn/extra.h"
#endif

std::string last_out;
std::map<int, float> occurences;

#ifdef __cplusplus
extern "C" {
#endif

rwkv_model_t rwkv_model_create(const char *path, const char *strategy) {
#ifdef FR_ENABLE_WEBRWKV
  if (std::string(strategy).substr(0, 6) == "webgpu") {
    is_webrwkv = true;
    init(time(NULL));
    try {
      if (std::string(path).find("ABC") != std::string::npos || 
        std::string(path).find("MIDI") != std::string::npos)
        load_with_rescale(path, 0, 0, 999);
      else
        load(path, 32, 32);
    } catch(...) {
      return nullptr;
    }
    return (void *)1;
  } else {
#else
  {
#endif
    rwkv_model_t handle;
    try {
      handle = new rwkv::Model(path, strategy);
    } catch(FRException &e) {
#ifdef __ANDROID__
      __android_log_print(ANDROID_LOG_ERROR, "faster-rwkv", "rwkv_model_create failed!");
      __android_log_print(ANDROID_LOG_ERROR, "faster-rwkv", "Error msg: %s", e.what());
#endif
      return nullptr;
    }
    return handle;
  }
}

rwkv_tokenizer_t rwkv_ABCTokenizer_create() {
  return new rwkv::ABCTokenizer();
}

rwkv_tokenizer_t rwkv_Tokenizer_create(const char *path) {
  return new rwkv::Tokenizer(path);
}

rwkv_sampler_t rwkv_sampler_create() {
  return new rwkv::Sampler();
}

void rwkv_sampler_set_seed(rwkv_sampler_t sampler_handle, int seed) {
  static_cast<rwkv::Sampler *>(sampler_handle)->set_seed(seed);
}

char rwkv_abcmodel_run_with_tokenizer_and_sampler(
    rwkv_model_t model_handle, rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle, const char input,
    // sampler params
    float temperature, int top_k, float top_p) {
  rwkv::ABCTokenizer *tokenizer =
      static_cast<rwkv::ABCTokenizer *>(tokenizer_handle);
  std::vector<int> input_id = tokenizer->encode(std::string(1, input));
  int output_id;
#ifdef FR_ENABLE_WEBRWKV
  if (is_webrwkv) {
    std::vector<uint16_t> input_ids_u16 = std::vector<uint16_t>(input_id.begin(), input_id.end());
    output_id = (int)infer(input_ids_u16.data(), input_ids_u16.size(), {temperature, top_p, static_cast<uintptr_t>(top_k)});
  } else {
#else
  {
#endif
    rwkv::Sampler *sampler = static_cast<rwkv::Sampler *>(sampler_handle);
    rwkv::Model *model = static_cast<rwkv::Model *>(model_handle);
    auto output_tensor = Copy(model->Run(input_id[0]), rwkv::Device::kCPU);
    output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
  }
  if (output_id == tokenizer->eos_token_id)
    return output_id;
  std::string output = tokenizer->decode(output_id);
  return output[0];
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
  std::string input_str(input, input_length);
  std::vector<int> input_ids = tokenizer->encode(input_str);
  int output_id;
#ifdef FR_ENABLE_WEBRWKV
  if (is_webrwkv) {
    std::vector<uint16_t> input_ids_u16 = std::vector<uint16_t>(input_ids.begin(), input_ids.end());
    output_id = (int)infer(input_ids_u16.data(), input_ids_u16.size(), {temperature, top_p, static_cast<uintptr_t>(top_k)});
  } else {
#else
  {
#endif
    rwkv::Sampler *sampler = static_cast<rwkv::Sampler *>(sampler_handle);
    rwkv::Model *model = static_cast<rwkv::Model *>(model_handle);
    for (int i = 0; i < input_ids.size(); i++) {
      if (i == (input_ids.size() - 1)) {
        auto output_tensor = Copy(model->Run(input_ids[i]), rwkv::Device::kCPU);
        output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
      } else {
        model->Run(input_ids[i]);
      }
    }
  }
  std::string output = tokenizer->decode(output_id);
  return (char)output[0];
}

char* rwkv_chatmodel_eval(
    rwkv_model_t model_handle, rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle,
    char *input,
    // sampler params
    float temperature, int top_k, float top_p,
    float presence_penalty, float frequency_penalty, float penalty_decay) {
  rwkv::Tokenizer *tokenizer =
      static_cast<rwkv::Tokenizer *>(tokenizer_handle);
  std::vector<int> input_id = tokenizer->encode(std::string(input));
  int output_id;
#ifdef FR_ENABLE_WEBRWKV
  if (is_webrwkv) {
    std::vector<uint16_t> input_ids_u16 = std::vector<uint16_t>(input_id.begin(), input_id.end());
    output_id = (int)infer(input_ids_u16.data(), input_ids_u16.size(), {temperature, top_p, static_cast<uintptr_t>(top_k)});
  } else {
#else
  {
#endif
    rwkv::Sampler *sampler = static_cast<rwkv::Sampler *>(sampler_handle);
    rwkv::Model *model = static_cast<rwkv::Model *>(model_handle);
    auto output_tensor = Copy(model->Run(input_id), rwkv::Device::kCPU);
    for (auto &[id, occurence] : occurences) {
      output_tensor.data_ptr<float>()[id] -=
          frequency_penalty * occurence + presence_penalty;
      occurence *= penalty_decay;
    }

    output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
    occurences[output_id]++;
  }
  if (output_id == 0) // end_of_sentense
    last_out = "<end>";
  else
    last_out = tokenizer->decode(output_id);
  return (char*)last_out.c_str();
}

int rwkv_model_load_states(rwkv_model_t model_handle, const char *path) {
  try {
    rwkv::Model *model = static_cast<rwkv::Model *>(model_handle);
    model->LoadStateFile(std::string(path));
  } catch(...) {
    return 1;
  }
  return 0;
}

void rwkv_model_clear_states(rwkv_model_t model_handle) {
#ifdef FR_ENABLE_WEBRWKV
  if (is_webrwkv) {
    clear_state();
  } else {
#else
  {
#endif
    static_cast<rwkv::Model *>(model_handle)->ResetStates();
    last_out.clear();
    occurences.clear();
  }
}

#ifdef __cplusplus
}
#endif
