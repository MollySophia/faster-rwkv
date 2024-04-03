#include <fstream>
#include <iostream>
#include <sstream>
#ifdef FR_ENABLE_ANDROID_ASSET
#include <android/asset_manager.h>
#endif

#include "librwkv-qualcomm.h"
#include "extra.h"
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#define private public
#include <model.h>
#undef private
#include <utils.h>

namespace rwkv {
namespace _qnn {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

void init_model(Model *model, Device device, const std::string &_path,
                const std::string &strategy, const std::any &extra) {

  auto [path, android_asset] = [&]() {
    if (_path.substr(0, 6) == "asset:") {
      return std::make_pair(_path.substr(6), true);
    }
    return std::make_pair(_path, false);
  }();

#ifndef FR_ENABLE_ANDROID_ASSET
  RV_CHECK(!android_asset);
#else
  if (android_asset) {
    RV_CHECK(extra.has_value());
  }
#endif

  auto remove_suffix = [](const std::string &str, const std::string &suffix) {
    if (str.size() < suffix.size()) {
      return str;
    }
    if (str.substr(str.size() - suffix.size()) == suffix) {
      return str.substr(0, str.size() - suffix.size());
    }
    return str;
  };

  bool context_binary = false;
  if (path.find(".bin") != std::string::npos) {
    context_binary = true;
  } else if (path.find(".so") != std::string::npos) {
    context_binary = false;
  } else {
    RV_UNIMPLEMENTED() << "unsupported model file: " << path << "\nExpecting .so or .bin";
  }

  path = remove_suffix(path, ".so");
  path = remove_suffix(path, ".bin");
  path = remove_suffix(path, ".config");

  const auto model_path = path + (context_binary ? ".bin" : ".so");
  const auto config_path = path + ".config";

  model->_extra = std::make_shared<QnnExtra>();
  auto &model_extra = *std::any_cast<std::shared_ptr<QnnExtra>>(model->extra());

  std::string config;
// #ifdef FR_ENABLE_ANDROID_ASSET
//   if (android_asset) {
//     auto *mgr = std::any_cast<AAssetManager *>(extra);
//     AAsset *asset =
//         AAssetManager_open(mgr, config_path.c_str(), AASSET_MODE_BUFFER);
//     if (asset) {
//       const char *config_data =
//           static_cast<const char *>(AAsset_getBuffer(asset));
//       auto config_size = AAsset_getLength(asset);
//       config = std::string(config_data, config_data + config_size);
//       AAsset_close(asset);
//     }
//   } else {
// #else
  {
// #endif
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

    model->_version = get_value("version");
    model->_act_dtype = str_to_dtype(get_value("act_dtype", "fp32"));
    model->_weight_dtype = str_to_dtype(get_value("weight_dtype", "fp32"));
    model->_head_size = std::stoi(get_value("head_size"));
    model->_n_embd = std::stoi(get_value("n_embd"));
    model->_head_size = model->_n_embd / model->_head_size;
    model->_n_layer = std::stoi(get_value("n_layer"));
    model->_n_att = std::stoi(get_value("n_att"));
    model->_n_ffn = std::stoi(get_value("n_ffn"));
    // model_extra.vocab_size = std::stoi(get_value("vocab_size"));
  }

  if (context_binary) {
// #if FR_ENABLE_ANDROID_ASSET
//     AAsset *asset = nullptr;
//     if (android_asset) {
//       auto *mgr = std::any_cast<AAssetManager *>(extra);
//       asset = AAssetManager_open(mgr, bin_path.c_str(), AASSET_MODE_BUFFER);
//       if (asset) {
//         const char *bin_data = static_cast<const char *>(AAsset_getBuffer(asset));
//         uint64_t bin_size = AAsset_getLength64(asset);
//         // todo
//       }
//     } else {
// #else
    {
// #endif
      QnnRwkvBackendCreateWithContext(&model_extra.backend, &model_extra.modelHandle, model_path, "libQnnHtp.so", "libQnnSystem.so");
    }
  } else {
    QnnRwkvBackendCreate(&model_extra.backend, &model_extra.modelHandle, model_path, "libQnnHtp.so");
  }

// #if FR_ENABLE_ANDROID_ASSET
//   if (asset) {
//     AAsset_close(asset);
//   } else {
// #else
  {
// #endif
    // todo
  }
}

KernelRegister init_model_reg("init_model", Device::kQNN, init_model);

} // namespace _qnn
} // namespace rwkv