#ifndef FASTER_RWKV_H
#define FASTER_RWKV_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* rwkv_model_t;
typedef void* rwkv_tokenizer_t;
typedef void* rwkv_sampler_t;
typedef void* rwkv_tensor_t;

/**
 * @brief Create an RWKV model.
 * 
 * @param path The path of the model.
 * @param strategy The strategy.
 * @return rwkv_model_t The handle to the created model.
 */
rwkv_model_t rwkv_model_create(const char* path, const char* strategy);

/**
 * @brief Create an RWKV ABCTokenizer.
 * 
 * @return rwkv_tokenizer_t The handle to the created ABCTokenizer.
 */
rwkv_tokenizer_t rwkv_ABCTokenizer_create();

// For chat and midi models
rwkv_tokenizer_t rwkv_Tokenizer_create(const char *path);

/**
 * @brief Create an RWKV sampler.
 * 
 * @return rwkv_sampler_t The handle to the created sampler.
 */
rwkv_sampler_t rwkv_sampler_create();

/**
 * @brief Set the seed of the sampler.
 * 
 * @param sampler_handle The handle to the sampler.
 * @param seed The seed.
 */
void rwkv_sampler_set_seed(rwkv_sampler_t sampler_handle, int seed);

/**
 * @brief Run the RWKV model with single input, with encoder/sampler/decoder.
 */
char rwkv_abcmodel_run_prompt(rwkv_model_t model_handle,
                    rwkv_tokenizer_t tokenizer_handle,
                    rwkv_sampler_t sampler_handle,
                    const char *input,
                    const int input_length,
                    // sampler params 
                    float temperature, int top_k, float top_p);

/**
 * @brief Run the RWKV model with single input, with encoder/sampler/decoder.
 */
char rwkv_abcmodel_run_with_tokenizer_and_sampler(rwkv_model_t model_handle,
                    rwkv_tokenizer_t tokenizer_handle,
                    rwkv_sampler_t sampler_handle,
                    const char input,
                    // sampler params 
                    float temperature, int top_k, float top_p);

void rwkv_model_clear_states(rwkv_model_t model_handle);

int rwkv_midimodel_check_stopped(rwkv_tokenizer_t tokenizer_handle);

char* rwkv_midimodel_run_prompt_from_file(rwkv_model_t model_handle,
  rwkv_tokenizer_t tokenizer_handle,
  rwkv_sampler_t sampler_handle,
  const char *input_path,
  const int input_path_length,
  // sampler params 
  float temperature, int top_k, float top_p
);

char* rwkv_midimodel_run_with_text_prompt(
  rwkv_model_t model_handle,
  rwkv_tokenizer_t tokenizer_handle,
  rwkv_sampler_t sampler_handle,
  const char *input_text,
  const int input_text_length,
  // sampler params 
  float temperature, int top_k, float top_p
);

char* rwkv_midimodel_run_with_tokenizer_and_sampler(rwkv_model_t model_handle,
  rwkv_tokenizer_t tokenizer_handle,
  rwkv_sampler_t sampler_handle,
  // sampler params
  float temperature, int top_k, float top_p);


void rwkv_midimodel_save_result_to_midi(const char *midi_path, const int midi_path_length);

#ifdef __cplusplus
}
#endif

#endif // FASTER_RWKV_H
