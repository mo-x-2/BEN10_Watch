#pragma once

#include <Arduino.h>

namespace FeatureProvider {

// Normalize window data in-place (mean subtraction per-axis)
void normalize(float* window, size_t frames, size_t channels);

// Flatten to model input (frames x channels -> flat)
void flatten(const float* window, size_t frames, size_t channels, float* out);

}



