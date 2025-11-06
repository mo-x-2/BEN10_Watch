#include "accelerometer_handler.h"

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

namespace {
Adafruit_MPU6050 g_mpu;
}

namespace AccelerometerHandler {

bool begin(uint8_t i2c_addr, int sda, int scl, uint32_t freq) {
  // Wireバスが既に初期化されている場合は、そのまま使用
  // 未初期化の場合は初期化（BEN10では既に初期化されている想定）
  if (!g_mpu.begin(i2c_addr, &Wire)) {
    return false;
  }
  g_mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  g_mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  return true;
}

bool read(AccelSample &out) {
  sensors_event_t a, g, temp;
  g_mpu.getEvent(&a, &g, &temp);
  out.ax = a.acceleration.x;
  out.ay = a.acceleration.y;
  out.az = a.acceleration.z;
  return true;
}

}  // namespace AccelerometerHandler



