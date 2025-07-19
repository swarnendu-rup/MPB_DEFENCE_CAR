import cv2
import requests
import time

# Configure ESP32 IP address
ESP32_IP = "http://192.168.1.XXX"  # Update with your ESP32's IP
GUN_DETECTED_URL = f"{ESP32_IP}/gun_detected"

# Initialize webcam (change index if needed)
cap = cv2.VideoCapture(0)

def detect_gun(frame):
    """
    Placeholder for gun detection logic.
    Replace with actual object detection implementation.
    """
    # Example dummy condition (replace with real detection)
    # This is just a placeholder using color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Simple red color detection (simulate gun detection)
    lower_red = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    upper_red = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_mask = cv2.bitwise_or(lower_red, upper_red)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return True if significant red region is detected
    # (This is just a placeholder demonstration)
    return any(cv2.contourArea(cnt) > 5000 for cnt in contours)

def send_gun_detected():
    """Send notification to ESP32 when gun is detected"""
    try:
        requests.post(GUN_DETECTED_URL)
        print("Gun detection notified to ESP32")
    except Exception as e:
        print(f"Error connecting to ESP32: {e}")

def main():
    print("Starting gun detection system...")
    frame_count = 0
    detection_cooldown = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        frame_count += 1
        
        # Process every 5 frames to reduce load
        if frame_count % 5 == 0:
            if detect_gun(frame):
                print("Gun detected! Enabling ESP32 button")
                send_gun_detected()
                detection_cooldown = 10  # Keep button enabled for next X frames
        
        # Display the frame (optional)
        cv2.imshow('Gun Detection Feed', frame)
        
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
