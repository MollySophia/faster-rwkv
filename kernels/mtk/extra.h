#include <string>
#include "rwkv_mtk.h"
#include "tensor.h"

struct MtkExtra {
  void* neuron_runtime;
  RWKVRuntimeOptions runtime_options;
  RWKVModelOptions model_options;
  int vocab_size;
  int n_chunks = 1;
  bool inited = false;
};

class MtkTensorWrapper {
public:
  MtkTensorWrapper(const rwkv::Tensor &tensor)
      : tensor(tensor) {}

  ~MtkTensorWrapper() = default;

  rwkv::Tensor tensor;
};
