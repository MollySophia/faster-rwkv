#include "faster_rwkvd.h"

#include "model.h"
#include "stdlib.h"
#include "sampler.h"
#include "tokenizer.h"
#include "tensor.h"
#include <fstream>

static void midi_to_str(const std::string &midi_path, std::string &result) {
  system(("midi_to_str.exe " + midi_path + " --output prompt.txt").c_str());
  std::ifstream ifs("prompt.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  result = buffer.str();
  result.replace(result.begin(), result.end(), "<start>", "<pad>");
  result.replace(result.begin(), result.end(), "<end>", "");
}

static void str_to_midi(const std::string &result, const std::string &midi_path) {
  std::string result_modified(result);
  result_modified.replace(result_modified.begin(), result_modified.end(), "<pad>", "<start>");
  std::ofstream ofs("result.txt");
  ofs << result_modified;
  system(("str_to_midi.exe --output " + midi_path + " result.txt").c_str());
}

#ifdef __cplusplus
extern "C" {
#endif

rwkv_model_t rwkv_model_create(const char* path, const char* strategy) {
  return new rwkv::Model(path, strategy);
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

char rwkv_abcmodel_run_prompt(rwkv_model_t model_handle,
                    rwkv_tokenizer_t tokenizer_handle,
                    rwkv_sampler_t sampler_handle,
                    const char *input,
                    const int input_length,
                    // sampler params 
                    float temperature, int top_k, float top_p) {
    rwkv::ABCTokenizer* tokenizer = static_cast<rwkv::ABCTokenizer*>(tokenizer_handle);
    rwkv::Sampler* sampler = static_cast<rwkv::Sampler*>(sampler_handle);
    rwkv::Model* model = static_cast<rwkv::Model*>(model_handle);
    std::string input_str(input, input_length);
    std::vector<int> input_ids = tokenizer->encode(input_str);
    input_ids.insert(input_ids.begin(), tokenizer->bos_token_id);
    int output_id;
    for (int i = 0; i < input_ids.size(); i++) {
      if (i == (input_ids.size() - 1)) {
        auto output_tensor = Copy(model->Run(input_ids[i]), rwkv::Device::kCPU);
        output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
      }
      else {
        model->Run(input_ids[i]);
      }
    }
    std::string output = tokenizer->decode(output_id);
    return (char)output[0];
}

void rwkv_model_clear_states(rwkv_model_t model_handle) {
  static_cast<rwkv::Model*>(model_handle)->ResetStates();
}

void rwkv_midimodel_run_prompt_from_file(rwkv_model_t model_handle,
                    rwkv_tokenizer_t tokenizer_handle,
                    rwkv_sampler_t sampler_handle,
                    const char *input_path,
                    const int input_path_length,
                    const char *output_path,
                    const int output_path_length,
                    const int max_length,
                    // sampler params 
                    float temperature, int top_k, float top_p) {
  rwkv::Tokenizer* tokenizer = static_cast<rwkv::Tokenizer*>(tokenizer_handle);
  rwkv::Sampler* sampler = static_cast<rwkv::Sampler*>(sampler_handle);
  rwkv::Model* model = static_cast<rwkv::Model*>(model_handle);
  std::string input_path_str(input_path, input_path_length);
  std::string output_path_str(output_path, output_path_length);
  std::string result;
  midi_to_str(input_path_str, result);

  std::vector<int> input_ids = tokenizer->encode(result);
  auto output_tensor = Copy(model->Run(input_ids), rwkv::Device::kCPU);
  std::map<int, float> occurences;
  for (int i = 0; i < max_length; i++) {
    for (const auto &[id, occurence] : occurences) {
      output_tensor.data_ptr<float>()[id] -= 0.5 * occurence;
    }
    output_tensor.data_ptr<float>()[0] += (i - 2000) / 500.;
    output_tensor.data_ptr<float>()[127] -= 1.;
    int output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
    for (const auto &[id, occurence] : occurences) {
      occurences[id] *= 0.997;
    }
    if (output_id >= 128 || output_id == 127) {
      occurences[output_id] += 1;
    } else {
      occurences[output_id] += 0.3;
    }
    if (output_id == tokenizer->eos_token_id()) {
      break;
    }
    std::string output = " " + tokenizer->decode(output_id);
    result += output;
    output_tensor = model->Run(output_id);
  }
  str_to_midi(result, output_path_str);
}

#ifdef __cplusplus
}
#endif
