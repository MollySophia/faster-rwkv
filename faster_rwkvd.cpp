#include "faster_rwkvd.h"

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

rwkv_model_t rwkv_model_create(const char* path, const char* strategy) {
  return new rwkv::Model(path, strategy);
}

rwkv_tokenizer_t rwkv_ABCTokenizer_create() {
  return new rwkv::ABCTokenizer();
}

rwkv_sampler_t rwkv_sampler_create() {
  return new rwkv::Sampler();
}

void rwkv_sampler_set_seed(rwkv_sampler_t sampler_handle, int seed) {
  static_cast<rwkv::Sampler*>(sampler_handle)->set_seed(seed);
}

char rwkv_abcmodel_run_with_tokenizer_and_sampler(rwkv_model_t model_handle,
                    rwkv_tokenizer_t tokenizer_handle,
                    rwkv_sampler_t sampler_handle,
                    const char input,
                    // sampler params 
                    float temperature, int top_k, float top_p) {
    rwkv::ABCTokenizer* tokenizer = static_cast<rwkv::ABCTokenizer*>(tokenizer_handle);
    rwkv::Sampler* sampler = static_cast<rwkv::Sampler*>(sampler_handle);
    rwkv::Model* model = static_cast<rwkv::Model*>(model_handle);
    std::vector<int> input_id = tokenizer->encode(std::string(1, input));
    auto output_tensor = Copy(model->Run(input_id[0]), rwkv::Device::kCPU);
    int output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
    std::string output = tokenizer->decode(output_id);
    return output[0];
}

#ifdef __cplusplus
}
#endif
