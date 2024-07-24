#include <fstream>
#include <iostream>
#include <chrono>

#include "extra.h"
#include "include/rwkv.h"
#include <kernels/graph_backend.h>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <model.h>
#include <tensor.h>

namespace rwkv {

template <> RwkvCppTensorWrapper Tensor::FromTensor() const {
  RV_CHECK(device() == Device::kCPU);
  return RwkvCppTensorWrapper(*this);
}

template <> Tensor Tensor::ToTensor(const RwkvCppTensorWrapper &wrapper) {
  return wrapper.tensor;
}

template <>
std::pair<RwkvCppTensorWrapper, std::vector<std::vector<RwkvCppTensorWrapper>>>
GraphBackendForwardInternal(const Model *model, int id,
                            std::vector<std::vector<RwkvCppTensorWrapper>> &&states) {
  auto &extra = *std::any_cast<std::shared_ptr<RwkvCppExtra>>(model->extra());

  size_t n_vocab = rwkv_get_n_vocab(extra.ctx);
  size_t state_size = rwkv_get_state_len(extra.ctx);

  float *state = new float[state_size];
  size_t elements_per_layer = states[0][0].tensor.numel() + states[0][2].tensor.numel() + states[0][1].tensor.numel();
  float *ptr = state;
  for (int i = 0; i < states.size(); i++) {
    memcpy(ptr, states[i][0].tensor.data_ptr<float>(), states[i][0].tensor.numel() * sizeof(float));
    memcpy(ptr + states[i][0].tensor.numel(), states[i][2].tensor.data_ptr<float>(), states[i][2].tensor.numel() * sizeof(float));
    memcpy(ptr + states[i][0].tensor.numel() + states[i][2].tensor.numel(), states[i][1].tensor.data_ptr<float>(), states[i][1].tensor.numel() * sizeof(float));
    ptr += elements_per_layer;
  }

  Shape output_shape;
  output_shape.push_back(n_vocab);
  RwkvCppTensorWrapper output(rwkv::Tensor::Empty(output_shape, DType::kFloat32, Device::kCPU));
  
  RV_CHECK(rwkv_eval(extra.ctx, id, state, state, output.tensor.data_ptr<float>()))
    << "rwkv_eval failed.";
  
  ptr = state;
  for (int i = 0; i < states.size(); i++) {
    memcpy(states[i][0].tensor.data_ptr<float>(), ptr, states[i][0].tensor.numel() * sizeof(float));
    memcpy(states[i][2].tensor.data_ptr<float>(), ptr + states[i][0].tensor.numel(), states[i][2].tensor.numel() * sizeof(float));
    memcpy(states[i][1].tensor.data_ptr<float>(), ptr + states[i][0].tensor.numel() + states[i][2].tensor.numel(), states[i][1].tensor.numel() * sizeof(float));
    ptr += elements_per_layer;
  }

  delete[] state;

  return {output, states};
}

Tensor RwkvCppModelForwardSeq(Model *model, Device device,
                       const std::vector<int> &ids,
                       bool full_output) {
  RV_CHECK(!full_output) << "full_output is not supported in fallback mode";
  // for (int i = 0; i < ids.size(); ++i) {
  //   auto id = ids[i];
  //   auto out = ModelForward(model, model->_act_device, id);
  //   if (i == ids.size() - 1) {
  //     return CopyToCPUIfAvailable(out);
  //   }
  // }

  auto &extra = *std::any_cast<std::shared_ptr<RwkvCppExtra>>(model->extra());
  auto &states = model->states();

  size_t n_vocab = rwkv_get_n_vocab(extra.ctx);
  size_t state_size = rwkv_get_state_len(extra.ctx);

  float *state = new float[state_size];
  size_t elements_per_layer = states[0][0].numel() + states[0][2].numel() + states[0][1].numel();
  float *ptr = state;
  for (int i = 0; i < states.size(); i++) {
    memcpy(ptr, states[i][0].data_ptr<float>(), states[i][0].numel() * sizeof(float));
    memcpy(ptr + states[i][0].numel(), states[i][2].data_ptr<float>(), states[i][2].numel() * sizeof(float));
    memcpy(ptr + states[i][0].numel() + states[i][2].numel(), states[i][1].data_ptr<float>(), states[i][1].numel() * sizeof(float));
    ptr += elements_per_layer;
  }

  Shape output_shape;
  output_shape.push_back(n_vocab);
  auto output = rwkv::Tensor::Empty(output_shape, DType::kFloat32, Device::kCPU);
  
  RV_CHECK(rwkv_eval_sequence(extra.ctx, (const uint32_t *)ids.data(), ids.size(), state, state, output.data_ptr<float>()))
    << "rwkv_eval failed.";
  
  ptr = state;
  for (int i = 0; i < states.size(); i++) {
    memcpy(states[i][0].data_ptr<float>(), ptr, states[i][0].numel() * sizeof(float));
    memcpy(states[i][2].data_ptr<float>(), ptr + states[i][0].numel(), states[i][2].numel() * sizeof(float));
    memcpy(states[i][1].data_ptr<float>(), ptr + states[i][0].numel() + states[i][2].numel(), states[i][1].numel() * sizeof(float));
    ptr += elements_per_layer;
  }

  delete[] state;

  return output;
}

KernelRegister RwkvCpp_model_forward_reg("model_forward", Device::kRwkvCpp,
                                      GraphBackendForward<RwkvCppTensorWrapper>);

KernelRegister RwkvCpp_model_forward_seq_reg("model_forward_seq", Device::kRwkvCpp,
                                       RwkvCppModelForwardSeq);

} // namespace rwkv