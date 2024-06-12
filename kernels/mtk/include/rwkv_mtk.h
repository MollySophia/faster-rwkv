#pragma once

#include "llm_types.h"

#include <string>
#include <vector>

typedef struct RWKVModelOptions {
    // Sizes
    size_t hiddenSize           = 2048;
    size_t vocabSize            = 65536;
    size_t numLayer             = 24;

    // Types
    LLMType embeddingType = LLMType::FP16;
    LLMType modelInputType  = LLMType::FP32;
    LLMType modelOutputType = LLMType::FP32;
} RWKVModelOptions;

typedef struct RWKVRuntimeOptions {
    std::string embPath;
    std::vector<std::string> dlaPaths;
    bool useModelBuffers = false;
    std::vector<void*> dlaBuffers;
    std::vector<size_t> dlaBufferSizes;
    void* embBuffer;
    size_t embBufferSize;
} RWKVRuntimeOptions;

bool neuron_rwkv_init(void** runtime, const RWKVModelOptions& modelOptions,
                       const RWKVRuntimeOptions& runtimeOptions);

void neuron_rwkv_release(void* runtime);

void* neuron_rwkv_inference_once(void* runtime, const int input_token);

void neuron_rwkv_reset(void* runtime);
