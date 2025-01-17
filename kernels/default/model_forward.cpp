#include <fstream>
#include <iostream>

#include <msgpack.hpp>

#include <kernels/export-ncnn/kernels.h>
#include <kernels/kernels.h>
#ifdef FR_ENABLE_ONNX
#include <kernels/export-onnx/kernels.h>
#endif
#include <kernels/registry.h>
#include <string>
#include <tensor.h>
#define private public
#include <model.h>
#undef private

namespace rwkv {

namespace def {

Tensor ModelForward(Model *model, Device device, int id) {
  auto &states = model->states();
  Tensor x = [&]() -> Tensor {
    if (model->_act_device == Device::kNCNNMeta
#ifdef FR_ENABLE_ONNX
        || model->_act_device == Device::kONNXMeta
#endif
    ) {
      Tensor embd_weights_cpu =
          Tensor::Empty({static_cast<long>(model->_embd_weights.size()),
                         model->_embd_weights[0].shape()[0]},
                        model->weight_dtype(), Device::kCPU);
      {
        auto fr_embd_dtype = model->_embd_weights[0].dtype();
        auto weight_dtype = model->weight_dtype();
        if (fr_embd_dtype == DType::kFloat16 &&
            weight_dtype == DType::kFloat32) {
          auto *ptr = embd_weights_cpu.data_ptr<float>();
          for (int i = 0; i < model->_embd_weights.size(); i++) {
            for (int j = 0; j < model->_n_embd; j++) {
              *ptr++ = model->_embd_weights[i].data_ptr<float16>()[j];
            }
          }
        } else if (fr_embd_dtype == DType::kFloat32 &&
                   weight_dtype == DType::kFloat32) {
          auto *ptr = embd_weights_cpu.data_ptr<float>();
          for (int i = 0; i < model->_embd_weights.size(); i++) {
            for (int j = 0; j < model->_n_embd; j++) {
              *ptr++ = model->_embd_weights[i].data_ptr<float>()[j];
            }
          }
        } else if (fr_embd_dtype == DType::kFloat16 &&
                   weight_dtype == DType::kFloat16) {
          auto *ptr = embd_weights_cpu.data_ptr<float16>();
          for (int i = 0; i < model->_embd_weights.size(); i++) {
            for (int j = 0; j < model->_n_embd; j++) {
              *ptr++ = model->_embd_weights[i].data_ptr<float16>()[j];
            }
          }
        } else {
          RV_UNIMPLEMENTED();
        }
      }
      if (model->_act_device == Device::kNCNNMeta) {
        Tensor id_tensor = ncnnmeta::add_input({1}, "input_id");
        for (int i = 0; i < states.size(); i++) {
          for (int j = 0; j < states[i].size(); j++) {
            auto state_name =
                "state_" + std::to_string(i) + "_" + std::to_string(j);
            auto &state_tensor = states[i][j];
            state_tensor =
                ncnnmeta::add_input(state_tensor.shape(), state_name);
          }
        }
        return ncnnmeta::Embedding(embd_weights_cpu, id_tensor);
      }
#ifdef FR_ENABLE_ONNX
      if (model->_act_device == Device::kONNXMeta) {
        Tensor input_id = onnxmeta::add_input({}, DType::kInt64, "input_id");

        Tensor embd_weights = onnxmeta::possible_initializer(embd_weights_cpu);
        return onnxmeta::gather(embd_weights, input_id);
      }
#endif
    }
    return model->_embd_weights[id];
  }();

  auto &params = model->_params;
#ifdef FR_ENABLE_ONNX
  if (model->_act_device == Device::kONNXMeta) {
    for (int i = 0; i < states.size(); i++) {
      for (int j = 0; j < states[i].size(); j++) {
        auto state_name =
            "state_" + std::to_string(i) + "_" + std::to_string(j);
        auto &state_tensor = states[i][j];
        state_tensor = onnxmeta::add_input(state_tensor.shape(),
                                           state_tensor.dtype(), state_name);
      }
    }
  }
#endif

  int param_idx = 0;
  Tensor v_first = Tensor::Empty({0}, DType::kFloat32, Device::kNCNNMeta);

  for (int i = 0; i < states.size(); ++i) {
    auto &state = states[i];

    {
      if (model->_version == "4") {
        std::tie(x, state[0], state[1], state[2], state[3]) = att(
            x, state[0], state[1], state[2], state[3], params[param_idx],
            params[param_idx + 1], params[param_idx + 2], params[param_idx + 3],
            params[param_idx + 4], params[param_idx + 5], params[param_idx + 6],
            params[param_idx + 7], params[param_idx + 8], params[param_idx + 9],
            params[param_idx + 10]);
        if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
          mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
          mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
          mark_as_output(state[2], "output_state_" + std::to_string(i) + "_2");
          mark_as_output(state[3], "output_state_" + std::to_string(i) + "_3");
        }
        param_idx += 11;
      } else if (model->_version == "5") {
        std::tie(x, state[0], state[1]) = att_one_v5(
            x, state[0], state[1], params[param_idx], params[param_idx + 1],
            params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
            params[param_idx + 5], params[param_idx + 6], params[param_idx + 7],
            params[param_idx + 8], params[param_idx + 9],
            params[param_idx + 10], params[param_idx + 11],
            params[param_idx + 12]);
        if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
          mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
          mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
        }
        param_idx += 13;
      } else if (model->_version == "5.1" || model->_version == "5.2") {
        std::tie(x, state[0], state[1]) = att_one_v5_1(
            x, state[0], state[1], params[param_idx], params[param_idx + 1],
            params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
            params[param_idx + 5], params[param_idx + 6], params[param_idx + 7],
            params[param_idx + 8], params[param_idx + 9],
            params[param_idx + 10], params[param_idx + 11],
            params[param_idx + 12], params[param_idx + 13],
            params[param_idx + 14]);
        if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
          mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
          mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
        }
        param_idx += 15;
      } else if (model->_version == "6") {
        std::tie(x, state[0], state[1]) = att_one_v6(
          x, state[0], state[1], params[param_idx], params[param_idx + 1],
            params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
            params[param_idx + 5], params[param_idx + 6], params[param_idx + 7],
            params[param_idx + 8], params[param_idx + 9],
            params[param_idx + 10], params[param_idx + 11],
            params[param_idx + 12], params[param_idx + 13],
            params[param_idx + 14], params[param_idx + 15], params[param_idx + 16],
            params[param_idx + 17], params[param_idx + 18], params[param_idx + 19],
            params[param_idx + 20]);
        if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
          mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
          mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
        }
        param_idx += 21;
      } else if (model->_version == "7") {
        if (i == 0) {
          std::tie(x, state[0], state[1], v_first) = att_one_v7(
            x, state[0], state[1], v_first, i,
            params[param_idx], params[param_idx + 1], // ln_w, ln_b
            params[param_idx + 2], params[param_idx + 3], // lx_w, lx_b
            params[param_idx + 4], params[param_idx + 5], // x_r, x_w
            params[param_idx + 6], params[param_idx + 7], // x_k, x_v
            params[param_idx + 8], params[param_idx + 9], // x_a, x_g
            params[param_idx + 10], params[param_idx + 11], params[param_idx + 12], // a0, a1, a2
            params[param_idx + 10], params[param_idx + 11], params[param_idx + 12], // v0, v1, v2
            params[param_idx + 13], params[param_idx + 14], params[param_idx + 15], // w0, w1, w2
            params[param_idx + 16], params[param_idx + 17], // g1, g2
            params[param_idx + 18], params[param_idx + 19], params[param_idx + 20], // k_k, k_a, r_k
            params[param_idx + 21], params[param_idx + 22], // kw, vw
            params[param_idx + 23], params[param_idx + 24] // rw, ow
          );
          param_idx += 25;
        } else {
          std::tie(x, state[0], state[1], v_first) = att_one_v7(
            x, state[0], state[1], v_first, i,
            params[param_idx], params[param_idx + 1], // ln_w, ln_b
            params[param_idx + 2], params[param_idx + 3], // lx_w, lx_b
            params[param_idx + 4], params[param_idx + 5], // x_r, x_w
            params[param_idx + 6], params[param_idx + 7], // x_k, x_v
            params[param_idx + 8], params[param_idx + 9], // x_a, x_g
            params[param_idx + 10], params[param_idx + 11], params[param_idx + 12], // a0, a1, a2
            params[param_idx + 13], params[param_idx + 14], params[param_idx + 15], // v0, v1, v2
            params[param_idx + 16], params[param_idx + 17], params[param_idx + 18], // w0, w1, w2
            params[param_idx + 19], params[param_idx + 20], // g1, g2
            params[param_idx + 21], params[param_idx + 22], params[param_idx + 23], // k_k, k_a, r_k
            params[param_idx + 24], params[param_idx + 25], // kw, vw
            params[param_idx + 26], params[param_idx + 27] // rw, ow
          );
          param_idx += 28;
        }
        if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
          mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
          mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
        }
      } else {
        RV_UNIMPLEMENTED();
      }
    }
    {
      int offset = 4;
      if (model->_version.substr(0, 1) != "4") {
        offset = 2;
      }

      if (model->_version == "7") {
        std::tie(x, state[offset]) = ffn_v7(
          x, state[offset], params[param_idx], params[param_idx + 1],
          params[param_idx + 2], params[param_idx + 3], params[param_idx + 4]);
        param_idx += 5;
      } else if (model->_version == "6") {
        std::tie(x, state[offset]) = ffn_v6(
          x, state[offset], params[param_idx], params[param_idx + 1],
          params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
          params[param_idx + 5], params[param_idx + 6]);
        param_idx += 7;
      } else {
        std::tie(x, state[offset]) = ffn(
          x, state[offset], params[param_idx], params[param_idx + 1],
          params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
          params[param_idx + 5], params[param_idx + 6]);
        param_idx += 7;
      }

      if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
        mark_as_output(state[offset], "output_state_" + std::to_string(i) +
                                          "_" + std::to_string(offset));
      }
    }

    if (x.dtype() == DType::kFloat16 && (i + 1) % model->_rescale_layer == 0) {
      scalar_div_(x, 2);
    }
  }

  //             x = F.layer_norm(x, (args.n_embd,),
  //             weight=w['ln_out.weight'], bias=w['ln_out.bias'])
  x = layernorm(x, params[param_idx], params[param_idx + 1]);

  //                 x = x @ w['head.weight']
  ncnnmeta::disable_int4(true);
  x = matmul(x, params[param_idx + 2]);
  ncnnmeta::disable_int4(false);
  if (x.dtype() == DType::kFloat16) {
    x = cast_dtype(x, DType::kFloat32);
  }
  if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
    mark_as_output(x, "output");
  }
  return x;
}

KernelRegister model_forward_reg_1("model_forward", Device::kCPU, ModelForward);
KernelRegister model_forward_reg_2("model_forward", Device::kCUDA,
                                   ModelForward);
KernelRegister model_forward_reg_3("model_forward", Device::kNCNNMeta,
                                   ModelForward);
KernelRegister model_forward_reg_4("model_forward", Device::kONNXMeta,
                                   ModelForward);

} // namespace def
} // namespace rwkv
