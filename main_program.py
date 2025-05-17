import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time
import os
from gpiozero import Button
from gpiozero import LED
from time import sleep
from signal import pause
from threading import Thread
from datetime import datetime
from picamera2 import Picamera2
import json
# Convert OpenCV frame to PIL
from PIL import Image
import numpy as np

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.model import CustomVGG
from utils.constants import INPUT_IMG_SIZE, NEG_CLASS

# Optional: use cv2.VideoCapture(0) if you're not on Raspberry Pi


gpio_input_trigger_pin = 23
gpio_output_pin = 21
# working_dir_path = r"C:\Users\gokul\Documents\projects\avo global wiper\Global_Wiper\anamoly_detection\\"
working_dir_path = r"C:\Users\gokul\Documents\projects\avo global wiper\global_wiper_final\\"
# Read path from file
with open("working_dir.txt", "r") as f:
    working_dir_path = f.read().strip()

# working_dir_path = r""
line_positions_json_file = working_dir_path+"line_positions.json"
cropped_image_path = working_dir_path+"cropped_images"
save_folder = working_dir_path+"captured_images"

os.makedirs(save_folder, exist_ok=True)
os.makedirs(cropped_image_path, exist_ok=True)

# Setup GPIO
gpio_trigger = Button(gpio_input_trigger_pin)
# Initialize pin 21 as an LED (can be used as output)
gpio_ouput = LED(gpio_output_pin)
gpio_input_trigger = False

# Setup Camera
picam2 = Picamera2()
# picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.size = (4056, 3040)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

#________________Annomaly Detection_______________________
# Load the trained model
device = torch.device("cpu")
model_path = working_dir_path+"weights/carbonbrush_model_2.h5"  # Update with your actual model path
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Define the preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(INPUT_IMG_SIZE),
    transforms.ToTensor(),
])

def list_files_in_directory(directory):
    # Get the list of files in the directory
    files = os.listdir(directory)
    cropped_image_list = []
    # Iterate through the list of files and print their full path
    for file in files:
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path):  # Check if it's a file (not a directory)
            cropped_image_list.append(full_path)
            # print(full_path)
    return cropped_image_list

def annomoly_detection():
    annomoly_detected = False
    cropped_image_list = list_files_in_directory(cropped_image_path)
    i = 1
    for path in cropped_image_list:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        # Measure start time
        start_time = time.time()

        # Run the model
        with torch.no_grad():
            probs, heatmap = model(img_tensor)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs).item()

        # Measure end time
        end_time = time.time()

        print("Brush No:"+str(i))
        i=i+1
        # Compute prediction time
        prediction_time = end_time - start_time
        print(f"Prediction Time: {prediction_time:.4f} seconds")

        # Display classification result
        label = "Good" if pred_class != NEG_CLASS else "Anomaly"
        print(f"Prediction: {label} (Confidence: {confidence:.2f})")
        if label=="Anomaly":
            annomoly_detected = True
            #break

    write_output(annomoly_detected)

def write_output(annomoly_detected):
    if annomoly_detected:
        print("Bad camper detected")
        gpio_ouput.on() #relay off
    else:
        print("Good camper detected")
        gpio_ouput.off() #relay on
    sleep(1)

#________________Capturing Image and crop_______________________
# Capture on GPIO trigger
def gpio_callback():
    global gpio_input_trigger
    gpio_input_trigger = True
    print("📶 GPIO 23 HIGH")
    print("Program triggered")

gpio_trigger.when_pressed = gpio_callback

def capture_image(frame):
    print("📸 Captured frame")
    image = opencv_to_pil(frame)
    # Launch crop UI
    root = tk.Tk()
    app = LineDrawer(root, image)
    app.save_crops()
    app.call_destroy()
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = "None"
    filename = os.path.join(save_folder, f"image_{timestamp}.jpg")
    print(f"📸 Capturing image: {filename}")
    image = picam2.capture_array()
    cv2.imwrite(filename, image)

class LineDrawer:
    def __init__(self, root, image, save_folder=cropped_image_path):
        self.root = root
        self.root.title("Crop Tool")

        self.original_image = image
        self.ori_width, self.ori_height = self.original_image.size

        self.display_width = 1280
        self.scale_ratio = self.display_width / self.ori_width
        self.display_height = int(self.ori_height * self.scale_ratio)

        self.display_image = self.original_image.resize(
            (self.display_width, self.display_height), Image.Resampling.LANCZOS
        )

        # Setup canvas with scroll
        outer_frame = tk.Frame(self.root)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(outer_frame, width=800, height=600, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scroll = tk.Scrollbar(outer_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        v_scroll = tk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.config(scrollregion=(0, 0, self.display_width, self.display_height))

        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.v_lines = [
            self.canvas.create_line(x, 0, x, self.display_height, fill="red", width=2)
            for x in [300, 1000]
        ]
        self.h_lines = [
            self.canvas.create_line(0, y, self.display_width, y, fill="blue", width=2)
            for y in range(0, self.display_height, self.display_height // 11)
        ]

        self.drag_data = {"line": None, "type": None}
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        self.save_button = tk.Button(self.root, text="💾 Save Crops", command=self.save_crops)
        self.save_button.pack(pady=10)

        self.save_folder = save_folder
        self.load_line_positions_from_json()

    def on_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        for line in self.v_lines:
            x0, _, x1, _ = self.canvas.coords(line)
            if abs(canvas_x - x0) < 10:
                self.drag_data = {"line": line, "type": "v"}
                return

        for line in self.h_lines:
            _, y0, _, y1 = self.canvas.coords(line)
            if abs(canvas_y - y0) < 10:
                self.drag_data = {"line": line, "type": "h"}
                return

    def on_drag(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        if self.drag_data["line"]:
            if self.drag_data["type"] == "v":
                self.canvas.coords(self.drag_data["line"], canvas_x, 0, canvas_x, self.display_height)
            elif self.drag_data["type"] == "h":
                self.canvas.coords(self.drag_data["line"], 0, canvas_y, self.display_width, canvas_y)

    def save_crops(self):
        x_disp = sorted([self.canvas.coords(line)[0] for line in self.v_lines])
        y_disp = sorted([self.canvas.coords(line)[1] for line in self.h_lines])

        x_positions = [int(x / self.scale_ratio) for x in x_disp]
        y_positions = [int(y / self.scale_ratio) for y in y_disp]

        image_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)

        for i in range(10):
            y1 = y_positions[i+1]
            y2 = y_positions[i+1 + 1]
            x1 = x_positions[0]
            x2 = x_positions[1]
            crop = image_cv[y1:y2, x1:x2]
            filename = os.path.join(self.save_folder, f"crop_{i+1}.png")
            cv2.imwrite(filename, crop)
        self.save_line_positions_to_json()

        print(f"✅ Saved 10 cropped images in '{self.save_folder}/'")
    
    def save_line_positions_to_json(self, filename=line_positions_json_file):
        data = {
            "v_lines": [round(self.canvas.coords(line)[0], 2) for line in self.v_lines],
            "h_lines": [round(self.canvas.coords(line)[1], 2) for line in self.h_lines],
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Saved line positions to {filename}")

    def load_line_positions_from_json(self, filename=line_positions_json_file):
        if not os.path.exists(filename):
            return  # No positions saved before

        with open(filename, "r") as f:
            data = json.load(f)

        try:
            for i, x in enumerate(data.get("v_lines", [])):
                self.canvas.coords(self.v_lines[i], x, 0, x, self.display_height)
            for i, y in enumerate(data.get("h_lines", [])):
                self.canvas.coords(self.h_lines[i], 0, y, self.display_width, y)
            print(f"✅ Loaded line positions from {filename}")
        except Exception as e:
            print(f"⚠️ Failed to load saved line positions: {e}")

    def call_destroy(self):
        self.root.destroy()
        
def opencv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#________________Camera Live Preview_______________________
def capture_from_camera():
    global gpio_input_trigger
    print("🎥 Press 'c' to capture a frame, or 'q' to quit.")
    while True:
        frame = picam2.capture_array()
        # Resize frame to 640x480
        resized_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Live Camera (Press 'c' to capture 'q' to quit)", resized_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C') or gpio_input_trigger:
            capture_image(frame)
            annomoly_detection()
            gpio_input_trigger = False
        elif key == ord('q') or key == ord('Q'):
            print("❌ Quit")
            picam2.stop()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    capture_from_camera()
