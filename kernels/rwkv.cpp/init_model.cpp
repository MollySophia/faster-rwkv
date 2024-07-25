#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "include/rwkv.h"
#include "extra.h"
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#define private public
#include <model.h>
#undef private
#include <utils.h>

#define TAG "faster-rwkv"

namespace rwkv {
namespace _RwkvCpp {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

void init_model(Model *model, Device device, const std::string &_path,
                const std::string &strategy, const std::any &extra) {

  auto [path, android_asset] = [&]() {
    if (_path.substr(0, 6) == "asset:") {
      return std::make_pair(_path.substr(6), true);
    }
    return std::make_pair(_path, false);
  }();

  RV_CHECK(!android_asset) << "Android asset is not supported yet for rwkv.cpp backend";

  auto remove_suffix = [](const std::string &str, const std::string &suffix) {
    if (str.size() < suffix.size()) {
      return str;
    }
    if (str.substr(str.size() - suffix.size()) == suffix) {
      return str.substr(0, str.size() - suffix.size());
    }
    return str;
  };

  path = remove_suffix(path, ".bin");
  path = remove_suffix(path, ".config");

  const auto config_path = path + ".config";

  model->_extra = std::make_shared<RwkvCppExtra>();
  auto &model_extra = *std::any_cast<std::shared_ptr<RwkvCppExtra>>(model->extra());

  std::string config;
  {
    std::ifstream config_file(config_path);
    if (config_file.good()) {
      std::stringstream ss;
      ss << config_file.rdbuf();
      config = ss.str();
    }
  }
  std::cout << config << std::endl;
  if (!config.empty()) {
    const auto get_value = [&config](const std::string &key,
                                     std::optional<std::string> default_value =
                                         std::nullopt) {
      const std::string key_with_colon = key + ": ";
      auto pos = config.find(key_with_colon);
      if (pos == std::string::npos) {
        if (default_value.has_value()) {
          return default_value.value();
        }
        RV_UNIMPLEMENTED() << "cannot find key: " << key
                           << " and default value is not provided";
      }
      pos += key_with_colon.size();
      auto pos2 = config.find("\n", pos);
      if (pos2 == std::string::npos) {
        pos2 = config.size();
      }
      return config.substr(pos, pos2 - pos);
    };
    const auto str_to_dtype = [](const std::string &str) {
      if (str == "fp32") {
        return DType::kFloat32;
      } else if (str == "fp16") {
        return DType::kFloat16;
      } else if (str == "int8") {
        return DType::kInt8;
      } else if (str == "int4") {
        return DType::kInt4;
      } else {
        RV_UNIMPLEMENTED() << "unsupported dtype: " << str;
      }
    };

    model->_version = get_value("version", "5.1");
    model->_act_dtype = str_to_dtype(get_value("act_dtype", "fp32"));
    model->_weight_dtype = str_to_dtype(get_value("weight_dtype", "fp32"));
    model->_head_size = std::stoi(get_value("head_size", "64"));
    model->_n_embd = std::stoi(get_value("n_embd", "512"));
    model->_head_size = model->_n_embd / model->_head_size;
    model->_n_layer = std::stoi(get_value("n_layer", "24"));
    model->_n_att = std::stoi(get_value("n_att", "512"));
    model->_n_ffn = std::stoi(get_value("n_ffn", "1792")); 
  } else {
    RV_UNIMPLEMENTED() << "No config file found";
  }

  model_extra.ctx = rwkv_init_from_file(_path.c_str(), 1, 0);
  RV_CHECK(model_extra.ctx != nullptr) << "Failed to init rwkv.cpp context from file: " << _path;
}

KernelRegister init_model_reg("init_model", Device::kRwkvCpp, init_model);

} // namespace _RwkvCpp
} // namespace rwkv