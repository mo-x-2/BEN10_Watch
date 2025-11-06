// ================== OLED / SERVO / ENCODER + MAX30102 (Dual I2C Buses) ==================
// Libraries
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>
#include "../alien.h"         // your bitmap set
#include <BluetoothSerial.h>

// ================== ML Inference (MagicWand) ==================
#include "accelerometer_handler.h"
#include "ring_buffer.h"
#include "feature_provider.h"
#include "model_settings.h"
#include "tflite_wrapper.h"
#include "magic_wand_model_data.h"

#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (70*1024)
#endif

// ML推論用のバッファとリングバッファ
static float g_ring_storage[ModelSettings::kWindowSize * ModelSettings::kChannelCount];
static FloatRingBuffer g_ring(g_ring_storage, ModelSettings::kWindowSize, ModelSettings::kChannelCount);
static float g_window[ModelSettings::kWindowSize * ModelSettings::kChannelCount];
static float g_input[ModelSettings::kInputElementCount];
static float g_scores[ModelSettings::kNumClasses];
static uint8_t g_tensor_arena[TENSOR_ARENA_SIZE];

// ML推論用のタイミング制御
unsigned long g_last_sample_ms = 0;
const unsigned long kSamplePeriodMs = 1000 / ModelSettings::kSampleRateHz;

// Bluetooth サーバ
BluetoothSerial BTServer;

// -------------------- OLED CONFIG (Bus: Wire on 21/22) --------------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET   -1
#define OLED_ADDR    0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// -------------------- PINS ---------------------------
const int ENC_PIN_A   = 25;   // EC11 CLK
const int ENC_PIN_B   = 26;   // EC11 DT
const int SWITCH_PIN  = 16;   // Limit switch: NO->pin, COM->GND (INPUT_PULLUP). OPEN=HIGH, CLOSED=LOW
const int SERVO_PIN   = 17;   // FS90 signal

// -------------------- SERVO CONFIG -------------------
Servo servo;
const int SERVO_OPEN_ANGLE  = 0;   // angle to hold while open
const int SERVO_CLOSE_ANGLE = 60;  // angle when closed
const int SERVO_MIN_US = 500;      // FS90 typical range
const int SERVO_MAX_US = 2500;

// -------------------- ENCODER STATE ------------------
volatile int32_t encSteps = 0;
volatile uint8_t prevAB   = 0;
const int STEPS_PER_SYMBOL = 8;

int8_t   symbolIndex        = 0;    // 0..3
int32_t  lastHandledSteps   = 0;
bool     systemEnabled      = false; // true only in OPEN_ACTIVE
bool     interruptsAttached = false;

// ====== STATE MACHINE ======
enum RunState { CLOSED_IDLE, OPEN_ACTIVE, COOLDOWN };
RunState state = CLOSED_IDLE;

const uint32_t COOLDOWN_MS = 10000; // 10 seconds
uint32_t cooldownStartMs = 0;

// ================== MAX30102 (Heart Rate) on SECOND I2C BUS ==================
#include "MAX30105.h"     // SparkFun MAX3010x library (supports MAX30102)
#include "heartRate.h"    // PBA beat detection

// Second I2C bus object (separate pins from OLED)
TwoWire I2C_HR = TwoWire(1);   // Use bus index 1 for the second I2C

MAX30105 sensor;

const byte RATE_SIZE = 8;      // moving average window
byte rates[RATE_SIZE] = {0};
byte rateSpot = 0;

uint32_t lastBeat = 0;         // ms timestamp of last beat
float    beatsPerMinute = 0;
int      beatAvg = 0;

bool hrAbove100 = false;       // evaluated from beatAvg
// ============================================================================

// -------------------- ENCODER ISR --------------------
IRAM_ATTR void isrEncoder() {
  uint8_t a = (uint8_t)digitalRead(ENC_PIN_A);
  uint8_t b = (uint8_t)digitalRead(ENC_PIN_B);
  uint8_t currAB = (a << 1) | b;
  uint8_t index  = ((prevAB << 2) | currAB) & 0x0F;

  if (index == 0b0001 || index == 0b0111 || index == 0b1110 || index == 0b1000) encSteps++;
  else if (index == 0b0010 || index == 0b1011 || index == 0b1101 || index == 0b0100) encSteps--;
  prevAB = currAB;
}

// -------------------- HELPERS ------------------------
void attachEncoderInterrupts() {
  if (!interruptsAttached) {
    prevAB = ((uint8_t)digitalRead(ENC_PIN_A) << 1) | (uint8_t)digitalRead(ENC_PIN_B);
    attachInterrupt(digitalPinToInterrupt(ENC_PIN_A), isrEncoder, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_PIN_B), isrEncoder, CHANGE);
    interruptsAttached = true;
  }
}

void detachEncoderInterrupts() {
  if (interruptsAttached) {
    detachInterrupt(digitalPinToInterrupt(ENC_PIN_A));
    detachInterrupt(digitalPinToInterrupt(ENC_PIN_B));
    interruptsAttached = false;
  }
}

void drawAlien(uint8_t idx) {
  display.clearDisplay();
  const uint8_t* bmp = (const uint8_t*)pgm_read_ptr(&ALIENS[idx % ALIEN_COUNT]);
  int x = (SCREEN_WIDTH - ALIEN_W) / 2;
  int y = (SCREEN_HEIGHT - ALIEN_H) / 2;
  display.drawBitmap(x, y, bmp, ALIEN_W, ALIEN_H, SSD1306_WHITE);
  display.display();
}

void turnDisplayOnAndShow() {
  display.ssd1306_command(SSD1306_DISPLAYON);
  drawAlien(symbolIndex);
}

void blankAndTurnDisplayOff() {
  display.clearDisplay();
  display.display(); // blank frame
  display.ssd1306_command(SSD1306_DISPLAYOFF);
}

void servoGoTo(int targetAngle) {
  int current = servo.read();
  int step = (targetAngle > current) ? 1 : -1;
  for (int a = current; a != targetAngle; a += step) {
    servo.write(a);
    delay(5); // ~300ms for 60°
  }
  servo.write(targetAngle);
}

// -------------------- HEART-RATE SAMPLING --------------------
void updateHeartRateFlag() {
  long irValue = sensor.getIR();

  // If no finger detected (very low signal), relax flag (state machine decides behavior)
  if (irValue < 5000) {
    hrAbove100 = false;
    return;
  }

  if (checkForBeat(irValue)) {
    uint32_t now = millis();
    uint32_t delta = now - lastBeat;
    lastBeat = now;

    if (delta > 0) {
      beatsPerMinute = 60.0f / (delta / 1000.0f);

      // Basic sanity filter
      if (beatsPerMinute > 20 && beatsPerMinute < 150) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;

        int sum = 0;
        for (byte i = 0; i < RATE_SIZE; i++) sum += rates[i];
        beatAvg = sum / RATE_SIZE;

        hrAbove100 = (beatAvg > 90);

        // Debug (optional)
        Serial.print(F("BPM: "));
        Serial.print(beatsPerMinute, 1);
        Serial.print(F("   Avg: "));
        Serial.print(beatAvg);
        Serial.print(F("   HighHR: "));
        Serial.println(hrAbove100 ? F("true") : F("false"));
      }
    }
  }
}

// -------------------- SETUP --------------------------
void setup() {
  Serial.begin(115200);

  // --- OLED bus (Wire) on SDA=21, SCL=22 ---
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // --- Heart sensor bus (I2C_HR) on SDA=18, SCL=19 ---
  // Change 18/19 if your board requires different pins for the second bus
  I2C_HR.begin(18, 19, 400000);

  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 init failed!");
    while (1) { delay(1000); }
  }

  pinMode(ENC_PIN_A, INPUT_PULLUP);
  pinMode(ENC_PIN_B, INPUT_PULLUP);
  pinMode(SWITCH_PIN, INPUT_PULLUP); // OPEN=HIGH, CLOSED=LOW

  // Servo init
  servo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);

  // MAX30102 init
  Serial.println(F("Initializing MAX30102..."));
  if (!sensor.begin(I2C_HR, I2C_SPEED_FAST)) {
    Serial.println(F("MAX30102 not found on secondary I2C. Check wiring/power."));
    // You can still run; watch will remain closed until HR is available
  } else {
    // Configure MAX30102 (tuned for fingertip)
    byte ledBrightness = 100; // 0–255 (~0–50mA)
    byte sampleAverage = 2;   // 1,2,4,8,16,32
    byte ledMode = 2;         // 1=Red, 2=Red+IR
    int  sampleRate = 200;    // Hz
    int  pulseWidth = 411;    // 69,118,215,411
    int  adcRange = 16384;    // 2048,4096,8192,16384
    sensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    // sensor.setPulseAmplitudeGreen(0); // if present and unused, keep off
    Serial.println(F("Place finger gently on the sensor."));
  }

  // MPU6050 (IMU) init for ML inference
  Serial.println(F("Initializing MPU6050..."));
  bool ok = AccelerometerHandler::begin(0x68, 21, 22, 400000);
  if (!ok) {
    Serial.println(F("MPU6050 init failed"));
    // ML inference will be skipped, but other functions continue
  } else {
    Serial.println(F("MPU6050 ready"));
  }

  // TFLite init
  Serial.println(F("Initializing TFLite..."));
  if (!TFLiteWrapper::init(g_magic_wand_model_data, g_magic_wand_model_data_len, g_tensor_arena, sizeof(g_tensor_arena))) {
    Serial.println(F("TFLite init failed (check model and arena size)"));
    // ML inference will be skipped, but other functions continue
  } else {
    Serial.println(F("TFLite ready"));
  }

  // Bluetooth サーバ開始
  BTServer.begin("ESP32_Server");
  Serial.println(F("Bluetooth Server Started (ESP32_Server)"));

  // Start CLOSED by assumption
  state = CLOSED_IDLE;
  systemEnabled = false;
  detachEncoderInterrupts();
  blankAndTurnDisplayOff();
  servo.write(SERVO_CLOSE_ANGLE);
}

// -------------------- ML INFERENCE --------------------
void updateMLInference() {
  unsigned long now = millis();
  if (now - g_last_sample_ms < kSamplePeriodMs) return;
  g_last_sample_ms = now;

  AccelSample s;
  if (!AccelerometerHandler::read(s)) return;
  float sample[ModelSettings::kChannelCount] = {s.ax, s.ay, s.az};
  g_ring.push(sample);

  if (!g_ring.full()) return;

  g_ring.latestWindow(g_window);
  FeatureProvider::normalize(g_window, ModelSettings::kWindowSize, ModelSettings::kChannelCount);
  FeatureProvider::flatten(g_window, ModelSettings::kWindowSize, ModelSettings::kChannelCount, g_input);

  // 推論実行
  char gestureChar = 'N';
  if (TFLiteWrapper::invoke(g_input, ModelSettings::kInputElementCount, g_scores, ModelSettings::kNumClasses)) {
    // ラベルに対応（0:W, 1:O, 2:L, 3:N）
    // 画像データに基づき、N以外のクラス（W/O/L）を検出するための統合閾値
    // W: 0.000-0.465, O: 0.001-0.072, L: 0.000-0.001 の範囲を考慮して閾値を0.05に設定
    const float kGestureThreshold = 0.05f;
    
    // N以外のクラスを優先的に検出（W/O/Lのスコアが閾値を超えているかチェック）
    // ベストクラスを決定し、そのスコアが閾値を超えていれば検出
    int best_class = 0;
    float best_score = g_scores[0];
    for (int i = 1; i < ModelSettings::kNumClasses - 1; ++i) { // N(3)を除くW/O/Lのみチェック
      if (g_scores[i] > best_score) {
        best_score = g_scores[i];
        best_class = i;
      }
    }
    
    // W/O/Lのいずれかが閾値を超えていれば検出
    if (best_score >= kGestureThreshold) {
      if (best_class == 0) gestureChar = 'W';
      else if (best_class == 1) gestureChar = 'O';
      else if (best_class == 2) gestureChar = 'L';
    } else {
      // 閾値未満の場合はN（通常動作）
      gestureChar = 'N';
    }
  }

  // Bluetoothで送信: "G:<W|O|L|N>;S:<0|1|2|3>\n"
  // G: ジェスチャー, S: symbolIndex
  if (BTServer.hasClient()) {
    BTServer.printf("G:%c;S:%d\n", gestureChar, symbolIndex);
  }
}

// -------------------- LOOP ---------------------------
void loop() {
  // Update HR (ignored during cooldown by state machine)
  updateHeartRateFlag();

  // Update ML inference (runs continuously)
  updateMLInference();

  bool switchOpen = (digitalRead(SWITCH_PIN) == HIGH); // HIGH=open, LOW=closed

  switch (state) {
    case CLOSED_IDLE: {
      // Wait for HR trigger (>100) to open
      if (hrAbove100) {
        systemEnabled = true;
        servoGoTo(SERVO_OPEN_ANGLE);  // open FIRST (mechanism pops; switch will open mechanically)
        encSteps = 0; lastHandledSteps = 0;
        attachEncoderInterrupts();
        turnDisplayOnAndShow();
        state = OPEN_ACTIVE;
      } else {
        // ensure off while closed
        systemEnabled = false;
        detachEncoderInterrupts();
        blankAndTurnDisplayOff();
        servo.write(SERVO_CLOSE_ANGLE);
      }
      break;
    }

    case OPEN_ACTIVE: {
      // Stay open regardless of HR until user closes watch (switch goes LOW)
      if (!switchOpen) {
        // Transition to COOLDOWN
        systemEnabled = false;
        detachEncoderInterrupts();
        blankAndTurnDisplayOff();
        servoGoTo(SERVO_CLOSE_ANGLE);

        // Reset HR averaging to avoid stale immediate retrigger
        for (byte i = 0; i < RATE_SIZE; i++) rates[i] = 0;
        rateSpot = 0;
        hrAbove100 = false;

        cooldownStartMs = millis();
        state = COOLDOWN;
      } else {
        // While open: handle encoder steps & update display
        noInterrupts();
        int32_t steps = encSteps;
        interrupts();

        int32_t delta = steps - lastHandledSteps;

        while (delta >= STEPS_PER_SYMBOL) {
          symbolIndex = (symbolIndex + 1) & 0x03;
          lastHandledSteps += STEPS_PER_SYMBOL;
          delta -= STEPS_PER_SYMBOL;
          drawAlien(symbolIndex);
        }
        while (delta <= -STEPS_PER_SYMBOL) {
          symbolIndex = (symbolIndex + 3) & 0x03; // (-1 mod 4)
          lastHandledSteps -= STEPS_PER_SYMBOL;
          delta += STEPS_PER_SYMBOL;
          drawAlien(symbolIndex);
        }
      }
      break;
    }

    case COOLDOWN: {
      // Everything off; ignore HR until cooldown ends
      systemEnabled = false;
      detachEncoderInterrupts();
      blankAndTurnDisplayOff();
      servo.write(SERVO_CLOSE_ANGLE);

      if (millis() - cooldownStartMs >= COOLDOWN_MS) {
        state = CLOSED_IDLE;  // ready for next cycle
      }
      break;
    }
  }

  delay(2);
}
