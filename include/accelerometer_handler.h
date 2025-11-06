// Accelerometer (MPU6050) abstraction
#pragma once

#include <Arduino.h>

struct AccelSample {
  float ax;
  float ay;
  float az;
};

namespace AccelerometerHandler {

bool begin(uint8_t i2c_addr = 0x68, int sda = I2C_SDA, int scl = I2C_SCL, uint32_t freq = 400000);
bool read(AccelSample &out);

}  // namespace AccelerometerHandler



