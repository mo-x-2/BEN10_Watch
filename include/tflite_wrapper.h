#pragma once

#include <Arduino.h>

namespace TFLiteWrapper {

bool init(const unsigned char* model_data, size_t model_size, uint8_t* tensor_arena, size_t arena_size);
bool invoke(const float* input, size_t input_len, float* output, size_t output_len);

}  // namespace TFLiteWrapper



