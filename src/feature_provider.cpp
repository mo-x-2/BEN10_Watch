#include "feature_provider.h"

namespace FeatureProvider {

void normalize(float* window, size_t frames, size_t channels) {
  // mean per channel
  for (size_t c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (size_t i = 0; i < frames; ++i) {
      sum += window[i * channels + c];
    }
    float mean = sum / static_cast<float>(frames);
    for (size_t i = 0; i < frames; ++i) {
      window[i * channels + c] -= mean;
    }
  }
}

void flatten(const float* window, size_t frames, size_t channels, float* out) {
  // already interleaved as [i*channels + c]
  size_t total = frames * channels;
  for (size_t i = 0; i < total; ++i) {
    out[i] = window[i];
  }
}

}



