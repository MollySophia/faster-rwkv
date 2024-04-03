#include <memory>
#include <vector>
#include "librwkv-qualcomm.h"
#include "tensor.h"
#include <cstring>

struct QnnExtra {
  QnnRwkvBackend_t backend;
  QnnRwkvModel_t modelHandle;
  rwkv::Shape output_shape;
};

class QnnTensorWrapper {
public:
  QnnTensorWrapper(const rwkv::Tensor &tensor)
      : tensor(tensor) {}

  ~QnnTensorWrapper() = default;

  rwkv::Tensor tensor;
};
