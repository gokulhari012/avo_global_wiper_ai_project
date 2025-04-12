from picamera2 import Picamera2
from datetime import datetime
import time
import os

# Create directory to store images if it doesn't exist 
working_dir_path = r"C:\Users\gokul\Documents\projects\avo global wiper\global_wiper_final\\"
# Read path from file
with open("global_wiper_final/working_dir.txt", "r") as f:
    working_dir_path = f.read().strip()
save_dir = working_dir_path+"captured_images"
os.makedirs(save_dir, exist_ok=True)

# Initialize the camera
picam2 = Picamera2()

# Configure camera
config = picam2.create_still_configuration()
picam2.configure(config)

# Start the camera
picam2.start()
time.sleep(2)  # Wait for camera to adjust

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{save_dir}/image_{timestamp}.jpg"

# Capture the image
picam2.capture_file(filename)

print(f"Image saved as: {filename}")

# Optional: stop the camera
picam2.close()
