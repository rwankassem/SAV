#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

const char *ssid = "Rwan";
const char *password = "sa45545sa";


constexpr uint32_t LOOP_DELAY_MS = 50;

void startCameraServer();

void setup()
{
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println();

  Serial.println("Configuring camera...");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound())
  {
    Serial.println("PSRAM found");
    config.frame_size = FRAMESIZE_QVGA; 
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  else
  {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }
  else
  {
    Serial.println("Camera initialized successfully");
  }

  // =========================
  // FIX ORIENTATION
  // =========================
  sensor_t * s = esp_camera_sensor_get();
  s->set_vflip(s, 1);  
  s->set_hmirror(s, 1);   

  Serial.println("Connecting to WiFi");

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  startCameraServer();

  Serial.println("Camera Ready!");
  Serial.print("Stream: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
}

void loop()
{
  delay(LOOP_DELAY_MS);
}