
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from PIL import Image
from utils.model import CustomVGG
from utils.constants import INPUT_IMG_SIZE, NEG_CLASS

working_dir_path = r"C:\Users\gokul\Documents\projects\avo global wiper\global_wiper_final\\"
# Read path from file
with open("global_wiper_final/working_dir.txt", "r") as f:
    working_dir_path = f.read().strip()

cropped_image_path = working_dir_path+"cropped_images"
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
    # print(cropped_image_list)
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

        # Compute prediction time
        print(i)
        prediction_time = end_time - start_time
        print(f"Prediction Time: {prediction_time:.4f} seconds")
        # Display classification result
        label = "Good" if pred_class != NEG_CLASS else "Anomaly"
        print(f"Prediction: {label} (Confidence: {confidence:.2f})")
        i=i+1
        if label == "Anomaly":
            annomoly_detected = True

    if annomoly_detected:
        print("\n\nFinally Anomally")
    else:
        print("\n\nFinally Good")

annomoly_detection()