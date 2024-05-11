#include <kernels/registry.h>
#include <tensor.h>
#include <chrono>

namespace rwkv {
namespace cpu {

inline float fast_exp(float x) {
  union {uint32_t i;float f;} v;
  v.i=(1<<23)*(1.4426950409*x+126.94201519f);
  return v.f;
}

Tensor softmax(const Tensor &x, float temperature) {
  Tensor y = Tensor::Empty(x.shape(), x.dtype(), x.device());
  auto *ptr = x.data_ptr<float>();
  auto *y_ptr = y.data_ptr<float>();
  int len = x.numel();
  const float max_logit = *std::max_element(ptr, ptr + len);
  float sum = 0;
  for (int i = 0; i < len; i++) {
    y_ptr[i] = fast_exp((ptr[i] - max_logit) / temperature);
    sum += y_ptr[i];
  }
  for (int i = 0; i < len; i++) {
    y_ptr[i] /= sum;
  }
  return y;
}

KernelRegister softmax_reg("softmax", Device::kCPU, softmax);

} // namespace cpu
} // namespace rwkv
