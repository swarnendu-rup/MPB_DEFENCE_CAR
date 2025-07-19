#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "AndroidAP56C4";
const char* password = "priya123";

// Removed unnecessary serverIP setting

const int ledPin = 2; // GPIO the LED is connected to
bool gunDetected = false;
unsigned long lastGunCheck = 0;

WebServer server(80);

void handleRoot() {
  String html = "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>ESP32 Gun Trigger</title>";
  html += "<style>button { padding: 20px; font-size: 24px; } button:disabled { cursor: not-allowed; }</style>";
  html += "</head><body><h2>ESP32 Gun Detection Trigger</h2>";
  html += "<button id='controlButton' disabled onclick='sendTrigger()'>LED Control</button>";
  html += "<script>";
  html += "function sendTrigger() { fetch('/trigger').then(response => response.text()).then(data => alert(data)); }";
  html += "setInterval(() => { fetch('/status').then(response => response.json()).then(data => {";
  html += "document.getElementById('controlButton').disabled = !data.triggerEnabled;";
  html += "document.getElementById('controlButton').style.cursor = data.triggerEnabled ? 'pointer' : 'not-allowed';";
  html += "}); }, 1000);";
  html += "</script></body></html>";
  server.send(200, "text/html", html);
}

void handleTrigger() {
  if (gunDetected) {
    digitalWrite(ledPin, !digitalRead(ledPin));
    server.send(200, "text/plain", "LED toggled");
  } else {
    server.send(403, "text/plain", "Trigger not enabled");
  }
}

void handleStatus() {
  String json = "{";
  json += "\"triggerEnabled\":" + String(gunDetected ? "true" : "false");
  json += "}";
  server.send(200, "application/json", json);
}

void handleGunDetected() {
  gunDetected = true;
  server.send(200, "text/plain", "Gun detection confirmed");
  lastGunCheck = millis();
}

void setup() {
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);

  Serial.begin(115200);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/trigger", HTTP_POST, handleTrigger);
  server.on("/status", handleStatus);
  server.on("/gun_detected", handleGunDetected);

  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
  if (millis() - lastGunCheck >= 5000) { // Reset after 5 seconds of no detection
    gunDetected = false;
  }
}
