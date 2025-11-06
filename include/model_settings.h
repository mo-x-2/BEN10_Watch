#pragma once

#include <Arduino.h>

namespace ModelSettings {

static const int kChannelCount = 3;      // ax, ay, az
static const int kSampleRateHz = 25;     // sampling rate
static const int kWindowSize = 40;       // frames per window
static const int kInputElementCount = kChannelCount * kWindowSize;

static const int kNumClasses = 4;        // e.g., W, O, L, negative

extern const char* kClassNames[kNumClasses];

}



