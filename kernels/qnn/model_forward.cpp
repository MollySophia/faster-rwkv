#include <fstream>
#include <iostream>

#include "extra.h"
#include "librwkv-qualcomm.h"
#include <kernels/graph_backend.h>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <model.h>
#include <tensor.h>

namespace rwkv {

// NOTE: the memory is shared here. You can also copy it if you want.
template <> QnnTensorWrapper Tensor::FromTensor() const {
  RV_CHECK(device() == Device::kCPU);
  return QnnTensorWrapper(*this);
}

// NOTE: the memory is not shared, otherwise the data may be released
template <> Tensor Tensor::ToTensor(const QnnTensorWrapper &wrapper) {
  return wrapper.tensor;
}

template <>
std::pair<QnnTensorWrapper, std::vector<std::vector<QnnTensorWrapper>>>
GraphBackendForwardInternal(const Model *model, int id,
                            std::vector<std::vector<QnnTensorWrapper>> &&states) {
  auto &extra = *std::any_cast<std::shared_ptr<QnnExtra>>(model->extra());

  QnnRwkvCopyStatesInPlace(extra.backend);
  QnnRwkvExecute(extra.backend, id);

  if (extra.output_shape.empty()) {
    std::vector<size_t> output_shape;
    QnnRwkvGetOutputShape(extra.backend, QnnRwkvGetOutputNum(extra.backend) - 1, output_shape);
    for (auto &s : output_shape) {
      extra.output_shape.push_back(s);
    }
  }

  QnnTensorWrapper output(rwkv::Tensor::Empty(extra.output_shape, DType::kFloat32, Device::kCPU));

  QnnRwkvGetOutput(extra.backend, QnnRwkvGetOutputNum(extra.backend) - 1, output.tensor.data_ptr<float>(), output.tensor.numel());

  return {output, states};
}

KernelRegister mtk_model_forward_reg("model_forward", Device::kQNN,
                                      GraphBackendForward<QnnTensorWrapper>);

} // namespace rwkv