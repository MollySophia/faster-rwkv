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

#ifdef _WIN32
#include "shellapi.h"
#include "windows.h"
#include <direct.h>
#endif

std::string result;
std::string last_out;
std::vector<int> token_ids;
std::map<int, float> occurences;

#ifdef _WIN32
static bool RunExec(const char *cmd, const char *para, DWORD ms) {
  SHELLEXECUTEINFO ShExecInfo = {0};
  ShExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
  ShExecInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
  ShExecInfo.hwnd = NULL;
  ShExecInfo.lpVerb = NULL;
  ShExecInfo.lpFile = cmd;
  ShExecInfo.lpParameters = para;
  ShExecInfo.lpDirectory = NULL;
  ShExecInfo.nShow = SW_HIDE;
  ShExecInfo.hInstApp = NULL;
  bool ret = ShellExecuteEx(&ShExecInfo);
  WaitForSingleObject(ShExecInfo.hProcess, ms);
  return ret;
}

static void midi_to_str(const char *midi_path, std::string &result) {
  char cwd[64];
  _getcwd(cwd, 64);
  std::cout << "Current Path: " << cwd << std::endl;
  char cmd[400];
  snprintf(cmd, sizeof(cmd), "%s --output lib\\fastmodel\\prompt.txt",
           midi_path);
  std::cout << "CMD: " << cmd << std::endl;
  std::ofstream ofs("lib\\fastmodel\\log.txt");
  ofs << "CMD: " << cmd << std::endl << "Current Path: " << cwd << std::endl;
  ofs.close();
  ofs.flush();
  // system(cmd);
  // WinExec(cmd, SW_HIDE);
  RunExec("lib\\fastmodel\\midi_to_str.exe", cmd, INFINITE);
  std::ifstream ifs("lib\\fastmodel\\prompt.txt");
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  result = buffer.str();
  if (result.substr(0, 7) == "<start>")
    result = "<pad>" + result.substr(7);
  if (result.substr(result.length() - 6) == " <end>")
    result = result.substr(0, result.length() - 6);
  std::cout << "midi_to_str end" << std::endl;
}

static void str_to_midi(const std::string &result, const char *midi_path) {
  std::string result_modified(result);
  result_modified = "<start>" + result_modified.substr(5) + " <end>";
  std::ofstream ofs("lib\\fastmodel\\result.txt");
  ofs << result_modified;
  ofs.flush();
  ofs.close();
  char cmd[400];
  snprintf(cmd, sizeof(cmd), "--output %s lib\\fastmodel\\result.txt",
           midi_path);
  RunExec("lib\\fastmodel\\str_to_midi.exe", cmd, INFINITE);
}
#else
static void midi_to_str(const char *midi_path, std::string &result) {
  std::cout << "midi_to_str is not implemented on this platform" << std::endl;
  return;
}

static void str_to_midi(const std::string &result, const char *midi_path) {
  std::cout << "str_to_midi is not implemented on this platform" << std::endl;
  return;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

rwkv_model_t rwkv_model_create(const char *path, const char *strategy) {
#ifdef FR_ENABLE_WEBRWKV
  if (std::string(strategy).substr(0, 6) == "webgpu") {
    is_webrwkv = true;
    init(time(NULL));
    load(path, 0, 0);
    return nullptr;
  } else {
#else
  {
#endif
    return new rwkv::Model(path, strategy);
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

const char* rwkv_chatmodel_eval_single(
    rwkv_model_t model_handle, rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle,
    const char *input,
    // sampler params
    float temperature, int top_k, float top_p) {
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
    auto output_tensor = Copy(model->Run(input_id[0]), rwkv::Device::kCPU);
    output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
  }
  std::string output = tokenizer->decode(output_id);
  return output.c_str();
}

const char* rwkv_chatmodel_eval_sequence(
    rwkv_model_t model_handle,
    rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle,
    const char *input,
    // sampler params
    float temperature, int top_k, float top_p) {
  rwkv::ABCTokenizer *tokenizer =
      static_cast<rwkv::ABCTokenizer *>(tokenizer_handle);
  std::string input_str(input);
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
  return output.c_str();
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
    token_ids.clear();
    last_out.clear();
    result.clear();
    occurences.clear();
  }
}

int rwkv_midimodel_check_stopped(rwkv_tokenizer_t tokenizer_handle) {
  auto tokenizer = static_cast<rwkv::Tokenizer *>(tokenizer_handle);
  if (token_ids.empty()) {
    std::cout << "rwkv_midimodel_check_stopped: token_ids is empty"
              << std::endl;
    return 1;
  }
  return (token_ids[token_ids.size() - 1] == tokenizer->eos_token_id()) ? 1 : 0;
}

char* rwkv_midimodel_run_with_text_prompt(rwkv_model_t model_handle,
                                         rwkv_tokenizer_t tokenizer_handle,
                                         rwkv_sampler_t sampler_handle,
                                         const char *input_text,
                                         const int input_text_length,
                                         // sampler params
                                         float temperature, int top_k,
                                         float top_p) {
  rwkv::Tokenizer *tokenizer = static_cast<rwkv::Tokenizer *>(tokenizer_handle);
  rwkv::Sampler *sampler = static_cast<rwkv::Sampler *>(sampler_handle);
  rwkv::Model *model = static_cast<rwkv::Model *>(model_handle);
  result = std::string(input_text, input_text_length);
  // std::cout << "Prompt: " << result << std::endl;
  std::vector<int> input_ids = tokenizer->encode(result);
  int output_id;
  for (int i = 0; i < input_ids.size(); i++) {
    if (i == (input_ids.size() - 1)) {
      auto output_tensor = Copy(model->Run(input_ids[i]), rwkv::Device::kCPU);
      output_tensor.data_ptr<float>()[0] -= 4; // ((int)token_ids.size() - 2000) / 500.0
      output_tensor.data_ptr<float>()[127] -= 1.;   // avoid "t125"
      output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
    } else {
      model->Run(input_ids[i]);
    }
  }
  last_out = tokenizer->decode(output_id);
  result += " " + last_out;
  token_ids.push_back(output_id);
  return (char *)last_out.c_str();
}

char* rwkv_midimodel_run_prompt_from_file(
    rwkv_model_t model_handle,
    rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle,
    const char *input_path,
    const int input_path_length,
    // sampler params
    float temperature, int top_k,
    float top_p) {
  std::cout << "midi input path: " << input_path << std::endl;
  std::string str;
  midi_to_str(input_path, str);
  return rwkv_midimodel_run_with_text_prompt(model_handle, tokenizer_handle,
                                      sampler_handle, str.c_str(), str.size(),
                                      temperature, top_k, top_p);
}

char* rwkv_midimodel_run_with_tokenizer_and_sampler(
    rwkv_model_t model_handle, rwkv_tokenizer_t tokenizer_handle,
    rwkv_sampler_t sampler_handle,
    // sampler params
    float temperature, int top_k, float top_p) {
  rwkv::Tokenizer *tokenizer = static_cast<rwkv::Tokenizer *>(tokenizer_handle);
  rwkv::Sampler *sampler = static_cast<rwkv::Sampler *>(sampler_handle);
  rwkv::Model *model = static_cast<rwkv::Model *>(model_handle);

  auto output_tensor =
      Copy(model->Run(token_ids[token_ids.size() - 1]), rwkv::Device::kCPU);
  for (const auto &[id, occurence] : occurences) {
    output_tensor.data_ptr<float>()[id] -= 0.5 * occurence;
  }
  if (!token_ids.empty())
    output_tensor.data_ptr<float>()[0] +=
        ((int)token_ids.size() - 2000) / 500.0; // not too short, not too long
  output_tensor.data_ptr<float>()[127] -= 1.;   // avoid "t125"
  int output_id = sampler->Sample(output_tensor, temperature, top_k, top_p);
  last_out = tokenizer->decode(output_id);
  result += " " + last_out;
  token_ids.push_back(output_id);
  for (const auto &[id, occurence] : occurences) {
    occurences[id] *= 0.997;
  }
  if (output_id >= 128 || output_id == 127) {
    occurences[output_id] += 1;
  } else {
    occurences[output_id] += 0.3;
  }
  return (char *)last_out.c_str();
}

void rwkv_midimodel_save_result_to_midi(const char *midi_path,
                                        const int midi_path_length) {
  str_to_midi(result, midi_path);
}

void rwkv_qualcomm_save_context(rwkv_model_t model_handle, const char *path) {
  // TODO
  return;
}

#ifdef __cplusplus
}
#endif
