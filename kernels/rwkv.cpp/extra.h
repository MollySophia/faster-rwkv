#include <string>
#include "include/rwkv.h"
#include "tensor.h"

struct RwkvCppExtra {
  struct rwkv_context *ctx;
};

class RwkvCppTensorWrapper {
public:
  RwkvCppTensorWrapper(const rwkv::Tensor &tensor)
      : tensor(tensor) {}

  ~RwkvCppTensorWrapper() = default;

  rwkv::Tensor tensor;
};
