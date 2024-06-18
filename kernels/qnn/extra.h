#include <string>
#include "librwkv-qualcomm.h"
#include "tensor.h"

struct QnnExtra {
  QnnRwkvBackend_t backend;
  QnnRwkvModel_t modelHandle;
  int vocab_size;
  rwkv::Shape output_shape;
  std::string model_path;
  std::string model_dir;
  std::string backend_str;
  bool context_binary;
};

class QnnTensorWrapper {
public:
  QnnTensorWrapper(const rwkv::Tensor &tensor)
      : tensor(tensor) {}

  ~QnnTensorWrapper() = default;

  rwkv::Tensor tensor;
};
