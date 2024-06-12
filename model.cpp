#include "model.h"

#include "check.h"
#include "kernels/kernels.h"
#include <tensor.h>
#include <utils.h>

#ifdef FR_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef FR_ENABLE_QNN
#include <kernels/qnn/include/librwkv-qualcomm.h>
#include <kernels/qnn/extra.h>
#endif
#ifdef FR_ENABLE_MTK
#include <kernels/mtk/include/rwkv_mtk.h>
#include <kernels/mtk/extra.h>
#endif
#include <fstream>
#include <iostream>
#include <msgpack.hpp>
#include <string>

namespace rwkv {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

Model::Model(const std::string &path, const std::string &strategy)
    : Model(path, strategy, std::any()) {}

Model::Model(const std::string &path, const std::string &strategy,
             std::any extra) {
  auto dev_str = strategy.substr(0, strategy.find(" "));
  Device act_device = [&]() {
    if (dev_str == "export-ncnn") {
      return Device::kNCNNMeta;
    } else if (dev_str == "export-onnx") {
      return Device::kONNXMeta;
    } else if (dev_str == "cuda") {
      return Device::kCUDA;
    } else if (dev_str == "cpu") {
      return Device::kCPU;
    } else if (dev_str == "ncnn") {
      return Device::kNCNN;
    } else if (dev_str == "onnx") {
      return Device::kONNX;
    } else if (dev_str == "qnn" || dev_str == "qualcomm") {
      return Device::kQNN;
    } else if (dev_str == "mtk") {
      return Device::kMTK;
    } else {
      RV_UNIMPLEMENTED();
    }
  }();
  _act_device = act_device;
  std::tie(_act_dtype, _weight_dtype) = [&]() -> std::pair<DType, DType> {
    std::string dtype_str = strategy.substr(strategy.find(" ") + 1);
    if (dtype_str == "int4") {
      return {DType::kFloat32, DType::kInt4};
    } else if (dtype_str == "int8") {
      return {DType::kFloat32, DType::kInt8};
    } else if (dtype_str == "fp16") {
      return {DType::kFloat16, DType::kFloat16};
    } else if (dtype_str == "fp32") {
      return {DType::kFloat32, DType::kFloat32};
    } else if (dtype_str == "auto") {
      // init them in backend
      return {DType::kUndefined, DType::kUndefined};
    } else {
      RV_UNIMPLEMENTED();
    }
  }();

  init_model(this, act_device, path, strategy, extra);
  if (kDebug) {
    std::cout << "Model inited" << std::endl;
    std::cout << "version: " << _version << std::endl;
    std::cout << "activation dtype: " << dtype_to_string(_act_dtype)
              << std::endl;
    std::cout << "weight dtype: " << dtype_to_string(_weight_dtype)
              << std::endl;
    std::cout << "head_size: " << _head_size << std::endl;
    std::cout << "n_embd: " << _n_embd << std::endl;
    std::cout << "n_layer: " << _n_layer << std::endl;
    std::cout << "n_att: " << _n_att << std::endl;
    std::cout << "n_ffn: " << _n_ffn << std::endl;
  }
  RV_CHECK(!_version.empty());
  RV_CHECK(_act_dtype != DType::kUndefined);
  RV_CHECK(_weight_dtype != DType::kUndefined);
  RV_CHECK(_n_layer > 0);
  RV_CHECK(_n_embd > 0);
  RV_CHECK(_n_att > 0);
  if (_version.substr(0, 1) == "5") {
    RV_CHECK(_head_size > 0);
    RV_CHECK(_n_ffn > 0);
  }
  ResetStates();
}

void Model::SaveStateFile(const std::string &path) {
  std::vector<std::vector<std::unordered_map<std::string, msgpack::object>>> mp_states;

  auto dtype_to_string_in_msgpack = [](DType dtype) {
    if (dtype == DType::kFloat32) {
      return "torch.float32";
    } else if (dtype == DType::kFloat16) {
      return "torch.float16";
    } else if (dtype == DType::kInt8) {
      return "torch.int8";
    } else {
      RV_UNIMPLEMENTED();
    }
  };
  msgpack::zone z;
  for (const auto& state : _states) {
    std::vector<std::unordered_map<std::string, msgpack::object>> mp_state;
    for (const auto& s : state) {
      std::unordered_map<std::string, msgpack::object> mp_s;
      std::vector<char> data_vec;
      data_vec.resize(s.numel() * s.elem_size());
      memcpy(data_vec.data(), s.data_ptr(), s.numel() * s.elem_size());
      mp_s["dtype"] = msgpack::object(dtype_to_string_in_msgpack(s.dtype()), z);
      mp_s["data"] = msgpack::object(data_vec, z);
      mp_s["shape"] = msgpack::object(s.shape(), z);
      mp_state.push_back(mp_s);
    }
    mp_states.push_back(mp_state);
  }

  std::ofstream ofs(path);
  msgpack::pack(ofs, mp_states);
}

void Model::LoadStateFile(const std::string &path) {
  return LoadStateFile(path, nullptr);
}

void Model::LoadStateFile(const std::string &path, void* asset_manager) {
  const std::string data = read_file(path, asset_manager);

  auto unpacker = msgpack::unpack(data.data(), data.length());
  auto obj = unpacker.get();
  auto states_mp = obj.as<std::vector<std::vector<msgpack::object>>>();
  RV_CHECK(states_mp.size() == _states.size());
  for (int i = 0; i < states_mp.size(); i++) {
    RV_CHECK(states_mp[i].size() == _states[i].size());
    for (int j = 0; j < states_mp[i].size(); j++) {
      const auto &state_mp = states_mp[i][j];
      auto new_state = Tensor::FromMsgPack(state_mp);
      RV_CHECK(new_state.shape() == _states[i][j].shape());
      _states[i][j] = new_state;
    }
  }
}

void Model::ResetStates() {
#ifdef FR_ENABLE_QNN
  if (_act_device == Device::kQNN) {
    QnnRwkvBackend_t _backend;
    auto &extra = *std::any_cast<std::shared_ptr<QnnExtra>>(this->extra());
    _backend = extra.backend;
    QnnRwkvResetStates(_backend);
    return;
  }
#endif
#ifdef FR_ENABLE_MTK
  if (_act_device == Device::kMTK) {
    auto &extra = *std::any_cast<std::shared_ptr<MtkExtra>>(this->extra());
    neuron_rwkv_reset(extra.neuron_runtime);
    return;
  }
#endif
  _states.clear();
  // TODO:
  auto device = (_act_device == Device::kNCNN || _act_device == Device::kONNX || _act_device == Device::kQNN || _act_device == Device::kMTK)
     ? Device::kCPU : _act_device;
  if (this->_version == "4") {
    for (int i = 0; i < _n_layer; i++) {
      _states.push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s1, 0), device));
      auto s2 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s3, 0), device));
      auto s4 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s4, -1e30), device));
      auto s5 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s5, 0), device));
    }
  } else {
    RV_CHECK(_version.substr(0, 1) == "5" || _version.substr(0, 1) == "6");
    for (int i = 0; i < _n_layer; i++) {
      _states.push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s1, 0), device));
      auto s2 = Tensor::Empty(Shape{this->_head_size, _n_att / this->_head_size,
                                    _n_embd / this->_head_size},
                              DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s3, 0), device));
    }
  }
}

static Tensor CopyToCPUIfAvailable(Tensor x) {
  // TODO: more elegant
  try {
    return Copy(x, Device::kCPU);
  } catch (std::exception &e) {
    return x;
  }
}

Tensor Model::Run(const std::vector<int> &ids) {
  if (kDebug) {
    std::cout << "[seq mode]Model::Run(";
    for (auto id : ids) {
      std::cout << id << ", ";
    }
    std::cout << ")" << std::endl;
  }
  if (ids.size() == 1) {
    return CopyToCPUIfAvailable(
        ModelForward(this, this->_act_device, ids[0]));
  } else {
    return CopyToCPUIfAvailable(
        ModelForwardSeq(this, this->_act_device, ids, false));
  }
}

Tensor Model::Run(int id) {
  return CopyToCPUIfAvailable(
      ModelForward(this, this->_act_device, id));
}

} // namespace rwkv
