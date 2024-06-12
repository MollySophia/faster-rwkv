#include <fstream>
#include <iostream>
#include <chrono>

#include "extra.h"
#include "rwkv_mtk.h"
#include <kernels/graph_backend.h>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <model.h>
#include <tensor.h>

namespace rwkv {

// NOTE: the memory is shared here. You can also copy it if you want.
template <> MtkTensorWrapper Tensor::FromTensor() const {
  RV_CHECK(device() == Device::kCPU);
  return MtkTensorWrapper(*this);
}

// NOTE: the memory is not shared, otherwise the data may be released
template <> Tensor Tensor::ToTensor(const MtkTensorWrapper &wrapper) {
  return wrapper.tensor;
}

template <>
std::pair<MtkTensorWrapper, std::vector<std::vector<MtkTensorWrapper>>>
GraphBackendForwardInternal(const Model *model, int id,
                            std::vector<std::vector<MtkTensorWrapper>> &&states) {
  auto &extra = *std::any_cast<std::shared_ptr<MtkExtra>>(model->extra());

  Shape output_shape = {extra.vocab_size};
  void* output_buffer = neuron_rwkv_inference_once(extra.neuron_runtime, id);
  MtkTensorWrapper output(rwkv::Tensor::FromPtr(output_buffer, output_shape, DType::kFloat32, Device::kCPU));

  return {output, states};
}

KernelRegister Mtk_model_forward_reg("model_forward", Device::kMTK,
                                      GraphBackendForward<MtkTensorWrapper>);

} // namespace rwkv