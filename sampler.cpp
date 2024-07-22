#include "sampler.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <chrono>

#include <check.h>
#include <kernels/kernels.h>
#include <tensor.h>

namespace rwkv {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

// hand-made distribution to get the same result on different platforms
// https://stackoverflow.com/questions/48730363/if-we-seed-c11-mt19937-as-the-same-on-different-machines-will-we-get-the-same
int distribution(std::vector<float> probs, std::minstd_rand0 &generator) {
  float sum = std::accumulate(probs.begin(), probs.end(), 0.f);
  float random_value = 1. * (generator() - generator.min()) /
                       (generator.max() - generator.min()) * sum;
  float cumsum = 0;
  for (int i = 0; i < probs.size(); i++) {
    cumsum += probs[i];
    if (cumsum >= random_value) {
      return i;
    }
  }
  RV_UNIMPLEMENTED();
}

Sampler::Sampler() {
  _generator.seed(std::random_device()());
}

int Sampler::Sample(const Tensor &logits, float temperature, int top_k,
                    float top_p) {
  if (kDebug) {
    std::cout << "Sample: temperature=" << temperature << ", top_k=" << top_k
              << ", top_p=" << top_p << std::endl;
  }

  size_t size = logits.numel();

  temperature = std::clamp(temperature, 0.1f, 5.f);
  if (top_k >= size)
    top_k = size;

  if (top_k == 0 || top_k == 1)
    return std::max_element(logits.data_ptr<float>(), logits.data_ptr<float>() + size) - logits.data_ptr<float>();

  // softmax
  float sum = 0;
  int *index = new int[size];
  float *probs = new float[size];

  const float max_logit = *std::max_element(logits.data_ptr<float>(), logits.data_ptr<float>() + size);

  for (int i = 0; i < size; i++) {
    probs[i] = std::exp(logits.data_ptr<float>()[i] - max_logit);
    sum += probs[i];
    index[i] = i;
  }

  if (top_k != size)
    std::nth_element(index, index + top_k,
          index + size,
          [&](int i, int j) { return probs[i] > probs[j]; });
    std::sort(index, index + top_k,
          [&](int i, int j) { return probs[i] > probs[j]; });

  int len = top_k;

  // top-p
  float cumsum = 0;
  for (int i = 0; i < len; i++) {
    probs[index[i]] /= sum;
    cumsum += probs[index[i]];
    if (cumsum >= top_p) {
      len = i + 1;
      break;
    }
  }

  // temperature
  if (fabs(temperature - 1.f) > 1e-6) {
    cumsum = 0;
    for (int i = 0; i < len; i++) {
      probs[index[i]] = std::pow(probs[index[i]], 1.f / temperature);
      cumsum += probs[index[i]];
    }
  }

  // random choice
  float random_value = 1. * (_generator() - _generator.min()) /
                      (_generator.max() - _generator.min()) * cumsum;
  
  int ret = -1;
  cumsum = 0;
  for (int i = 0; i < len; i++) {
    cumsum += probs[index[i]];
    if (cumsum >= random_value) {
      ret = index[i];
      delete[] index;
      delete[] probs;
      return ret;
    }
  }
  
  delete[] index;
  delete[] probs;
  RV_UNIMPLEMENTED();
}

void Sampler::set_seed(int seed) { _generator.seed(seed); }

} // namespace rwkv
