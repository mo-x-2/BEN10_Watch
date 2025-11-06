#include "tflite_wrapper.h"

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

namespace {
tflite::MicroErrorReporter g_error_reporter;
const tflite::Model* g_model = nullptr;
tflite::AllOpsResolver g_resolver;  // simple: include all ops
std::unique_ptr<tflite::MicroInterpreter> g_interpreter;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
}

namespace TFLiteWrapper {

bool init(const unsigned char* model_data, size_t model_size, uint8_t* tensor_arena, size_t arena_size) {
  g_model = tflite::GetModel(model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    return false;
  }
  g_interpreter.reset(new tflite::MicroInterpreter(g_model, g_resolver, tensor_arena, arena_size, &g_error_reporter));
  if (!g_interpreter) return false;
  if (g_interpreter->AllocateTensors() != kTfLiteOk) return false;
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  return (g_input && g_output);
}

bool invoke(const float* input, size_t input_len, float* output, size_t output_len) {
  if (!g_input || !g_output) return false;

  // Handle input types (float or int8)
  if (g_input->type == kTfLiteFloat32) {
    float* in_ptr = g_input->data.f;
    for (size_t i = 0; i < input_len; ++i) in_ptr[i] = input[i];
  } else if (g_input->type == kTfLiteInt8) {
    const float scale = g_input->params.scale;
    const int zero_point = g_input->params.zero_point;
    int8_t* in_ptr = g_input->data.int8;
    for (size_t i = 0; i < input_len; ++i) {
      const int32_t q = static_cast<int32_t>(roundf(input[i] / scale)) + zero_point;
      in_ptr[i] = static_cast<int8_t>(max<int32_t>(-128, min<int32_t>(127, q)));
    }
  } else {
    return false;
  }

  if (g_interpreter->Invoke() != kTfLiteOk) return false;

  // Handle output types (float or int8), always return dequantized float scores
  if (g_output->type == kTfLiteFloat32) {
    float* out_ptr = g_output->data.f;
    for (size_t i = 0; i < output_len; ++i) output[i] = out_ptr[i];
  } else if (g_output->type == kTfLiteInt8) {
    const float scale = g_output->params.scale;
    const int zero_point = g_output->params.zero_point;
    int8_t* out_ptr = g_output->data.int8;
    for (size_t i = 0; i < output_len; ++i) {
      output[i] = (static_cast<int>(out_ptr[i]) - zero_point) * scale;
    }
  } else {
    return false;
  }
  return true;
}

}  // namespace TFLiteWrapper


