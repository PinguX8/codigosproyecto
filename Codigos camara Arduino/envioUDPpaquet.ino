#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>

// =========================
// WIFI
// =========================
const char* ssid = "Equipo 7";
const char* password = "Equipo07!";

// =========================
// UDP
// =========================
WiFiUDP udp;
const char* udpAddress = "192.168.100.14";
const int udpPort = 12345;
uint16_t frame_id = 0;
const size_t CHUNK_SIZE = 1000;

// =========================
// AI THINKER PINS
// =========================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void setup() {
  Serial.begin(115200);

  // WIFI
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado");
  Serial.print("IP ESP32: ");
  Serial.println(WiFi.localIP()); // ← ver IP en monitor serial

  // CONFIG CAMARA
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // OPTIMIZACION
  config.frame_size   = FRAMESIZE_QVGA; // 320x240
  config.jpeg_quality = 10;             // un poco mejor calidad
  config.fb_count     = 2;             // doble buffer

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Error camara: 0x%x\n", err);
    return;
  }

  Serial.println("Camara lista. Enviando frames...");
}

void loop() {

  // Reconexión automática
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi perdido, reconectando...");
    WiFi.reconnect();
    delay(3000);
    return;
  }

  // CAPTURAR FOTO
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Error capturando frame");
    return;
  }

  // ENVIAR POR UDP EN CHUNKS
  size_t img_len = fb->len;
  uint8_t* img_buf = fb->buf;
  uint16_t total_chunks = (img_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
  uint8_t packet_buffer[CHUNK_SIZE + 8];

  for (uint16_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
    size_t offset       = chunk_idx * CHUNK_SIZE;
    size_t payload_size = min((size_t)CHUNK_SIZE, img_len - offset);

    // Header 8 bytes
    packet_buffer[0] = (frame_id >> 8) & 0xFF;
    packet_buffer[1] =  frame_id       & 0xFF;
    packet_buffer[2] = (total_chunks >> 8) & 0xFF;
    packet_buffer[3] =  total_chunks       & 0xFF;
    packet_buffer[4] = (chunk_idx >> 8) & 0xFF;
    packet_buffer[5] =  chunk_idx       & 0xFF;
    packet_buffer[6] = (payload_size >> 8) & 0xFF;
    packet_buffer[7] =  payload_size       & 0xFF;

    memcpy(&packet_buffer[8], &img_buf[offset], payload_size);

    udp.beginPacket(udpAddress, udpPort);
    udp.write(packet_buffer, payload_size + 8);
    udp.endPacket();

    delayMicroseconds(50);
  }

  frame_id++;
  esp_camera_fb_return(fb);

  delay(100); // ~10 FPS (antes era 200ms = 5 FPS)
}