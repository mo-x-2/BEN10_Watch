#pragma once

#include <Arduino.h>

class FloatRingBuffer {
 public:
  FloatRingBuffer(float* storage, size_t capacity, size_t channels)
      : storage_(storage), capacity_(capacity), channels_(channels), count_(0), head_(0) {}

  void push(const float* sample) {
    size_t base = head_ * channels_;
    for (size_t c = 0; c < channels_; ++c) {
      storage_[base + c] = sample[c];
    }
    head_ = (head_ + 1) % capacity_;
    if (count_ < capacity_) count_++;
  }

  bool full() const { return count_ == capacity_; }

  void latestWindow(float* out) const {
    // Copy in chronological order from oldest to newest
    size_t start = (head_ + capacity_ - count_) % capacity_;
    for (size_t i = 0; i < count_; ++i) {
      size_t idx = (start + i) % capacity_;
      size_t base = idx * channels_;
      for (size_t c = 0; c < channels_; ++c) {
        out[i * channels_ + c] = storage_[base + c];
      }
    }
  }

  size_t size() const { return count_; }
  size_t capacity() const { return capacity_; }
  size_t channels() const { return channels_; }

 private:
  float* storage_;
  size_t capacity_;
  size_t channels_;
  size_t count_;
  size_t head_;
};



