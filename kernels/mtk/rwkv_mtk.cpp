#include "rwkv_mtk.h"
#include "fcntl.h"
#include <chrono>

#include "runtime/neuron_runtime.h"

#include "executor/rwkv_executor.h"
// #include "executor/tflite_executor.h"

#include "utils/dump.h"
#include "utils/logging.h"
#include "utils/half.hpp"

#define LLM_API __attribute__((visibility("default")))

static constexpr bool kUseUsdkBackend = true;

using RWKVDlaExecutor = RWKVExecutor<NeuronUsdkExecutorSingleUserIO>;

class SimpleEmb {
    public:
        SimpleEmb(std::string embPath, LLMType embFileType, LLMType outputType, size_t embDim, size_t vocabSize)
            : mEmbPath(embPath), mEmbType(embFileType), mOutputType(outputType), mEmbDim(embDim), mVocabSize(vocabSize) {
                int fd = open(embPath.c_str(), O_RDONLY);
                if (fd < 0) {
                    LOG(FATAL) << "Failed to open embedding file: " << embPath;
                    return;
                }

                size_t outElemSize = getLLMTypeSize(outputType); 
                size_t fileElemSize = getLLMTypeSize(embFileType);
                buffer = mmap(nullptr, embDim * vocabSize * fileElemSize, PROT_READ, MAP_PRIVATE, fd, 0);
                if (buffer == MAP_FAILED) {
                    LOG(FATAL) << "Failed to mmap embedding file: " << embPath;
                    return;
                }

                output.buffer = malloc(embDim * outElemSize);
                if (output.buffer == nullptr) {
                    LOG(FATAL) << "Failed to allocate output buffer";
                    return;
                }
                output.sizeBytes = embDim * outElemSize;
                output.usedSizeBytes = embDim * outElemSize;
            }
        
        ~SimpleEmb() {
            if (buffer != nullptr && buffer != MAP_FAILED) {
                munmap(buffer, mEmbDim * mVocabSize * getLLMTypeSize(mEmbType));
            }

            if (output.buffer != nullptr) {
                free(output.buffer);
                output.sizeBytes = 0;
                output.usedSizeBytes = 0;
            }
        }

        void runInference(const int inputToken) {
            size_t embSize = getLLMTypeSize(mEmbType);
            size_t outSize = getLLMTypeSize(mOutputType);
            size_t offset = inputToken * mEmbDim * embSize;
            if (mOutputType == mEmbType) {
                memcpy(output.buffer, (char*)buffer + offset, mEmbDim * embSize);
            }
            else if(mEmbType == LLMType::FP16 && mOutputType == LLMType::FP32) {
                float* ptr = (float*)output.buffer;
                half_float::half *halfPtr = (half_float::half*)((char*)buffer + offset);
                for (size_t i = 0; i < mEmbDim; i++) {
                    ptr[i] = half_float::half_cast<float>(halfPtr[i]);
                }
            }
            else {
                LOG(FATAL) << "Unsupported yet";
                // TODO
            }
        }

        IOBuffer& getOutput() {
            return output;
        }

    private:
        std::string mEmbPath;
        LLMType mEmbType;
        LLMType mOutputType;
        size_t mEmbDim;
        size_t mVocabSize;
        void* buffer = nullptr;
        IOBuffer output;
};

struct RWKVRuntime {
    std::vector<Executor*> dlaExecutors;
    SimpleEmb* simpleEmbExecutor;
    // Executor* tfliteEmbExecutor;
    RWKVRuntimeOptions options;
};

bool LLM_API neuron_rwkv_init(void** runtime, const RWKVModelOptions& modelOptions,
                               const RWKVRuntimeOptions& runtimeOptions) {

    if constexpr (kUseUsdkBackend) {
        LOG(DEBUG) << "Using NeuronUsdk (NeuronAdapter)";
    } else {
        LOG(DEBUG) << "Using Neuron Runtime";
        if (!init_neuron_runtime_library()) {
            LOG(ERROR) << "Failed to initialize runtime library.";
            *runtime = nullptr;
            return false;
        }
    }
    if (!init_dmabuf_library()) {
        LOG(ERROR) << "Failed to initialize dmabuf library.";
        *runtime = nullptr;
        return false;
    }

    const auto& dlaChunkPaths = runtimeOptions.dlaPaths;
    const auto numChunk = dlaChunkPaths.size();

    // Create rwkv runtime
    RWKVRuntime* rwkvRuntime = new RWKVRuntime;
    rwkvRuntime->options = runtimeOptions;

    for (int chunkIdx = 0; chunkIdx < numChunk; ++chunkIdx) {
        RWKVRuntimeInfo runtimeInfo;
        runtimeInfo.modelPath = dlaChunkPaths[chunkIdx];
        runtimeInfo.n_layer = modelOptions.numLayer / numChunk;
        LOG(DEBUG) << "Loading DLA " << chunkIdx;

        auto dlaExec = new RWKVDlaExecutor(runtimeInfo);
        rwkvRuntime->dlaExecutors.push_back(dlaExec);
    }

    LOG(DEBUG) << "Loading Embedding Weights: " << runtimeOptions.embPath;
    rwkvRuntime->simpleEmbExecutor = new SimpleEmb(runtimeOptions.embPath,
                                                   modelOptions.embeddingType,
                                                   modelOptions.modelOutputType,
                                                   modelOptions.hiddenSize,
                                                   modelOptions.vocabSize);
    LOG(DEBUG) << "Initialized Embedding Weights";

    // Chain the IO between the runtime chunks:
    // InputToken -> [Embedding -> DlaChunk1 -> DlaChunk2 -> ... -> DlaChunkN]-> Output
    auto getPrevChunkOutput = [&](const int curChunkIdx) -> const IOBuffer& {
        // First chunk gets the output from the embedding runtime
        return (curChunkIdx == 0)
            ? rwkvRuntime->simpleEmbExecutor->getOutput()
            : rwkvRuntime->dlaExecutors[curChunkIdx - 1]->getOutput();
    };
    for (int chunkIdx = 0; chunkIdx < numChunk; ++chunkIdx) {
        // Initialize after setModelInput so that the buffer allocator doesn't need to allocate for
        // inputs that are using an existing buffer.
        auto dlaExec = rwkvRuntime->dlaExecutors[chunkIdx];
        if (chunkIdx != 0)
            dlaExec->setModelInput(0, getPrevChunkOutput(chunkIdx));
        dlaExec->initialize(); // load model and allocate buffers
        dlaExec->registerRuntimeIO(); // Attach allocated buffers to model IO
        LOG(DEBUG) << "Initialized DLA " << chunkIdx;
    }

    *runtime = rwkvRuntime;
    return true;
}

void LLM_API neuron_rwkv_release(void* runtime) {
    auto rwkvRuntime = reinterpret_cast<RWKVRuntime*>(runtime);
    for (auto dlaExec : rwkvRuntime->dlaExecutors) {
        dlaExec->release();
        delete dlaExec;
    };
    rwkvRuntime->dlaExecutors.clear();
    delete rwkvRuntime->simpleEmbExecutor;
    delete rwkvRuntime;
}

void* LLM_API neuron_rwkv_inference_once(void* runtime, const int input_token) {
    auto rwkvRuntime = reinterpret_cast<RWKVRuntime*>(runtime);

    rwkvRuntime->simpleEmbExecutor->runInference(input_token);
    auto first_input_buffer = rwkvRuntime->dlaExecutors.front()->getInput();
    memcpy(first_input_buffer.buffer, rwkvRuntime->simpleEmbExecutor->getOutput().buffer, first_input_buffer.sizeBytes);

    size_t chunkIdx = 0;
    for (auto dlaExec : rwkvRuntime->dlaExecutors) {
        auto rwkvDlaExec = static_cast<RWKVDlaExecutor*>(dlaExec);
        rwkvDlaExec->runInference();
        chunkIdx++;
    }

    // Return logits
    const auto finalExecutor = rwkvRuntime->dlaExecutors.back();
    auto logitsBuffer = finalExecutor->getOutputBuffer();
    return reinterpret_cast<char*>(logitsBuffer);
}

void LLM_API neuron_rwkv_reset(void* runtime) {
    auto rwkvRuntime = reinterpret_cast<RWKVRuntime*>(runtime);
    for (auto dlaExec : rwkvRuntime->dlaExecutors) {
        auto rwkvDlaExec = static_cast<RWKVDlaExecutor*>(dlaExec);
        rwkvDlaExec->resetStates();
    }
}