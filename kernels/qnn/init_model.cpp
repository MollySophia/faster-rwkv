#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#ifdef FR_ENABLE_ANDROID_ASSET
#include <android/asset_manager.h>
#include <android/log.h>
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

#define TAG "faster-rwkv"

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
  std::string library_path;

#ifndef _WIN32
#ifdef FR_ENABLE_ANDROID_ASSET
  if (android_asset) {
    setenv("ADSP_LIBRARY_PATH", path.substr(path.find_last_of(":") + 1).c_str(), 1);
    path = path.substr(0, path.find_last_of(":"));
  } 
  else 
#endif
  {
    if (path.find_last_of(":") != std::string::npos) {
      setenv("ADSP_LIBRARY_PATH", path.substr(path.find_last_of(":") + 1).c_str(), 1);
      library_path = path.substr(path.find_last_of(":") + 1);
      path = path.substr(0, path.find_last_of(":"));
    } else {
      const auto model_dir = path.substr(0, path.find_last_of("/") + 1);
      setenv("ADSP_LIBRARY_PATH", model_dir.c_str(), 1);
      library_path = model_dir;
    }
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
  } else if (path.find(".so") != std::string::npos || path.find(".dll") != std::string::npos) {
    context_binary = false;
  } else {
    RV_UNIMPLEMENTED() << "unsupported model file: " << path << "\nExpecting .so or .bin";
  }

  path = remove_suffix(path, ".so");
  path = remove_suffix(path, ".dll");
  path = remove_suffix(path, ".bin");
  path = remove_suffix(path, ".config");
  path = remove_suffix(path, ".emb");

#ifdef _WIN32
  const auto model_path = path + (context_binary ? ".bin" : ".dll");
#else
  const auto model_path = path + (context_binary ? ".bin" : ".so");
#endif
  const auto config_path = path + ".config";
  const auto emb_path = path + ".emb";

  model->_extra = std::make_shared<QnnExtra>();
  auto &model_extra = *std::any_cast<std::shared_ptr<QnnExtra>>(model->extra());

  std::string config;
#ifdef FR_ENABLE_ANDROID_ASSET
  if (android_asset) {
    auto *mgr = std::any_cast<AAssetManager *>(extra);
    AAsset *asset =
        AAssetManager_open(mgr, config_path.c_str(), AASSET_MODE_BUFFER);
    if (asset) {
      const char *config_data =
          static_cast<const char *>(AAsset_getBuffer(asset));
      auto config_size = AAsset_getLength(asset);
      config = std::string(config_data, config_data + config_size);
      AAsset_close(asset);
    }
  } else {
#else
  {
#endif
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
    model_extra.vocab_size = std::stoi(get_value("vocab_size"));
    model_extra.backend_str = get_value("qnn_backend", "HTP");
  } else { // default config for abcmusic
    if (model_path.find("ABC") == std::string::npos) {
      RV_UNIMPLEMENTED() << "No config file found";
    }

    if (model_path.find("RWKV-6") != std::string::npos) {
      model->_version = "6";
    } else {
      model->_version = "5.1";
    }
    model->_act_dtype = DType::kFloat32;
    model->_weight_dtype = DType::kFloat32;
    model->_head_size = 64;
    model->_n_embd = 512;
    model->_n_layer = 24;
    model->_n_att = 512;
    model->_n_ffn = 1792;
  }

#ifdef FR_ENABLE_ANDROID_ASSET
  AAsset *asset = nullptr;
  AAsset *emb_asset = nullptr;
#endif

  if (context_binary) {
    {

#ifdef _WIN32
      if (StatusCode::SUCCESS != QnnRwkvBackendCreateWithContext(&model_extra.backend, &model_extra.modelHandle, model_path, "QnnHtp.dll", "QnnSystem.dll")) {
          RV_UNIMPLEMENTED() << "QnnRwkvBackendCreateWithContext failed";
      }
#else
      std::string backend_lib = "libQnnHtp.so";
      std::string system_lib = "libQnnSystem.so";
      if (!library_path.empty()) {
        backend_lib = library_path + "/" + backend_lib;
        system_lib = library_path + "/" + system_lib;
      }
      if (StatusCode::SUCCESS != QnnRwkvBackendCreateWithContext(&model_extra.backend, &model_extra.modelHandle, model_path, backend_lib, system_lib)) {
        RV_UNIMPLEMENTED() << "QnnRwkvBackendCreateWithContext failed";
      }
#endif
    }
  } else {
#ifdef _WIN32
      std::string backend_lib;
      if (model_extra.backend_str == "HTP") {
        backend_lib = "QnnHtp.dll";
      } else if (model_extra.backend_str == "GPU") {
        backend_lib = "QnnGpu.dll";
      } else if (model_extra.backend_str == "CPU") {
        backend_lib = "QnnCpu.dll";
      } else {
        RV_UNIMPLEMENTED() << "unsupported backend: " << model_extra.backend_str;
      }

      if (StatusCode::SUCCESS != QnnRwkvBackendCreate(&model_extra.backend, &model_extra.modelHandle, model_path, backend_lib)) {
        if (model_extra.backend_str == "HTP") {
          backend_lib = "QnnGpu.dll";
          model_extra.backend_str = "GPU";
          if (StatusCode::SUCCESS != QnnRwkvBackendCreate(&model_extra.backend, &model_extra.modelHandle, model_path, backend_lib)) {
            backend_lib = "QnnCpu.dll";
            model_extra.backend_str = "CPU";
            if (StatusCode::SUCCESS != QnnRwkvBackendCreate(&model_extra.backend, &model_extra.modelHandle, model_path, backend_lib)) {
              RV_UNIMPLEMENTED() << "QnnRwkvBackendCreate failed";
            }
          }
        } else {
          RV_UNIMPLEMENTED() << "QnnRwkvBackendCreate failed";
        }
      }
#else
      std::string backend_lib;
      backend_lib = "libQnnHtp.so";
      if (!library_path.empty()) {
        backend_lib = library_path + "/" + backend_lib;
      }

#ifdef FR_ENABLE_ANDROID_ASSET
      __android_log_print(ANDROID_LOG_INFO, TAG, "Trying qualcomm %s backend", backend_lib.c_str());
#endif
      if (StatusCode::SUCCESS != QnnRwkvBackendCreate(&model_extra.backend, &model_extra.modelHandle, model_path, backend_lib)) {
          RV_UNIMPLEMENTED() << "QnnRwkvBackendCreate failed";
      }
#endif

    std::string model_dir = model_path.substr(0, path.find_last_of("/") + 1);
    std::cout << "model_path: " << model_path << std::endl;
    std::cout << "model_dir: " << model_dir << std::endl;
    QnnRwkvSaveContext(model_extra.backend, model_dir);
    if (config.find("HTP") != std::string::npos) {
      config.replace(config.find("HTP"), 3, model_extra.backend_str);
    }
    std::ofstream config_file(model_dir + "/model_cache.config");
    config_file << config;
  }

#ifdef FR_ENABLE_ANDROID_ASSET
  if (asset) {
    AAsset_close(asset);
  } else {
#else
  {
#endif

  }
}

KernelRegister init_model_reg("init_model", Device::kQNN, init_model);

} // namespace _qnn
} // namespace rwkv