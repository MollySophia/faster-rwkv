#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

// ./midi_music <tokenizer> <model> <strategy>
// Example: ./midi_music midi_tokenizer midi_model "ncnn fp16"
int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  rwkv::Tokenizer tokenizer(argv[1]);
  rwkv::Sampler sampler;
  rwkv::Model model(argv[2], argv[3]);

  std::string input =
      "<pad> p:2c:a t4 p:2c:0 t62 p:2c:a t4 p:2c:0 t62 p:2c:a t4 p:2c:0 t62 p:2c:a t4 p:2c:0 t62 p:2a:a";
  // std::string input = "<pad>";

  std::vector<int> input_ids = tokenizer.encode(input);
  std::cout << "Prompt length: " << input_ids.size() << std::endl;

  std::cout << input;
  static const int N_TRIAL = 1;
  static const int length = 4096;
  int i = 1;
  for (int t = 0; t < N_TRIAL; t++) {
    std::string result;
    auto start = std::chrono::system_clock::now();
    auto output_tensor = Copy(model.Run(input_ids), rwkv::Device::kCPU);
    std::map<int, float> occurences;
    for (i = 1; i < length; i++) {
      // translated from ChatRWKV
      for (const auto &[id, occurence] : occurences) {
        output_tensor.data_ptr<float>()[id] -= 0.5 * occurence;
      }
      output_tensor.data_ptr<float>()[0] +=
          (i - 2000) / 500.;                      // not too short, not too long
      output_tensor.data_ptr<float>()[127] -= 1.; // avoid "t125"

      auto output_id = sampler.Sample(
                                // output_tensor, 1.f, 100, 1.f);
                                output_tensor, 5.f, 100, 0.6f);

      // translated from ChatRWKV
      for (const auto &[id, occurence] : occurences) {
        occurences[id] *= 0.997;
      }
      if (output_id >= 128 || output_id == 127) {
        occurences[output_id] += 1;
      } else {
        occurences[output_id] += 0.3;
      }

      if (output_id == tokenizer.eos_token_id()) {
        std::cout << " <end>";
        break;
      }
      std::string output = " " + tokenizer.decode(output_id);
      std::cout << output;
      result += output;
      output_tensor = model.Run(output_id);
    }
    auto end = std::chrono::system_clock::now();
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (kShowSpeed) {
      std::cout << "time: " << total_time.count() << "ms" << std::endl;
      std::cout << "num tokens: " << i + input_ids.size() << std::endl;
      std::cout << "ms per token: " << 1. * total_time.count() / (i + input_ids.size())
                << std::endl;
      std::cout << "tokens per second: "
                << 1. * (i + input_ids.size()) / total_time.count() * 1000 << std::endl;
    }
    const std::string filename = "midi_" + std::to_string(t) + ".txt";
    std::ofstream ofs(filename);
    // the str_to_midi.py requires <start> & <end> as separator
    result = "<start> " + result + " <end>";
    ofs << result;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Saved to " << filename << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
