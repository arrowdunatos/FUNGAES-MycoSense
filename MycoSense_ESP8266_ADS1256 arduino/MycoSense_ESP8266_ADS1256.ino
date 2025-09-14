#include <Arduino.h>
#include <SPI.h>
#include "ADS1256.h"

const int PIN_SCK    = 14;
const int PIN_MISO   = 12;
const int PIN_MOSI   = 13;
const int PIN_CS     = 15;
const int PIN_DRDY   = 5;
const int PIN_RESET  = 4;

const int NODES = 4;
const unsigned long SAMPLE_PERIOD_MS = 10;

ADS1256 adc;

void setup() {
  Serial.begin(115200);
  delay(200);

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS);

  if (!adc.begin(PIN_CS, PIN_DRDY, PIN_RESET)) {
    Serial.println("ADC initialization failed");
    while (true) delay(1000);
  }

  adc.setGain(1);
  adc.setDataRate(3000);
  delay(200);

  Serial.println("ADC ready");
}

inline unsigned long nowMs() {
  return millis();
}

void loop() {
  for (int ch = 0; ch < NODES; ++ch) {
    adc.setChannel(ch);
    adc.waitDRDY();
    float volts = adc.readCurrentChannelVolts();
    unsigned long ts = nowMs();

    Serial.print(ts);
    Serial.print(',');
    Serial.print(ch);
    Serial.print(',');
    Serial.println(volts, 6);

    delay(SAMPLE_PERIOD_MS);
  }
}
