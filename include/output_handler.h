#pragma once

#include <Arduino.h>

namespace OutputHandler {

void printScores(const char* const* labels, size_t num_labels, const float* scores);

}



