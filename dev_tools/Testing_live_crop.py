
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from PIL import Image
from utils.model import CustomVGG
from utils.constants import INPUT_IMG_SIZE, NEG_CLASS


image_path = r"C:\Users\gokul\Documents\projects\avo global wiper\rashpi datas\testing\testing\bottom3good\captured_image_18.jpg"

device = torch.device("cpu")
model_path = "weights/carbonbrush_model_2.h5"  # Update with your actual model path
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize(INPUT_IMG_SIZE),
    transforms.ToTensor(),
])


# === Original (full-res) crop size in pixels ===
ORIG_RECT_WIDTH = 150
ORIG_RECT_HEIGHT = 100
DISPLAY_WIDTH = 1000
SAVE_PATH = "D:/localized defect and heatmap/imgs2/saved_region"
ext=".png"

# === Global Variables ===
rect_x, rect_y = 50, 50  # top-left corner of rectangle (in display coords)
dragging = False
offset_x, offset_y = 0, 0
original_image = None
resized_image = None
scale_x, scale_y = 1.0, 1.0
display_rect_w, display_rect_h = 100, 100  # Will be calculated

def mouse_events(event, x, y, flags, param):
    global rect_x, rect_y, dragging, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_x <= x <= rect_x + display_rect_w and rect_y <= y <= rect_y + display_rect_h:
            dragging = True
            offset_x = x - rect_x
            offset_y = y - rect_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            rect_x = x - offset_x
            rect_y = y - offset_y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def draw_rectangle(img):
    return cv2.rectangle(img.copy(), (rect_x, rect_y), 
                         (rect_x + display_rect_w, rect_y + display_rect_h), 
                         (0, 255, 0), 2)

def main():
    global original_image, resized_image
    global scale_x, scale_y, display_rect_w, display_rect_h
    count=0
    # Load image

    original_image = cv2.imread(image_path)

    if original_image is None:
        print("❌ Could not load image.")
        return

    h, w = original_image.shape[:2]

    # Resize image for display
    if w > DISPLAY_WIDTH:
        scale = DISPLAY_WIDTH / w
        display_w = int(w * scale)
        display_h = int(h * scale)
        resized_image = cv2.resize(original_image, (display_w, display_h))
    else:
        resized_image = original_image.copy()
        scale = 1.0

    scale_x = original_image.shape[1] / resized_image.shape[1]
    scale_y = original_image.shape[0] / resized_image.shape[0]

    # Rectangle size in display image
    display_rect_w = int(ORIG_RECT_WIDTH / scale_x)
    display_rect_h = int(ORIG_RECT_HEIGHT / scale_y)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_events)

    while True:
        display_img = draw_rectangle(resized_image)
        cv2.imshow("Image", display_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Convert display rect to original image coordinates
            orig_x = int(rect_x * scale_x)
            orig_y = int(rect_y * scale_y)

            # Crop region from original image
            roi = original_image[orig_y:orig_y + ORIG_RECT_HEIGHT, orig_x:orig_x + ORIG_RECT_WIDTH]

            if roi.shape[0] > 0 and roi.shape[1] > 0:
                #cv2.imwrite(SAVE_PATH+str(count)+ext, roi)
                #count=count+1
                #print(f"✅ Saved region to {SAVE_PATH}")
                roi_pil = Image.fromarray(roi)
                img_tensor = transform(roi_pil).unsqueeze(0).to(device)
                start_time = time.time()
                with torch.no_grad():
                    probs, heatmap = model(img_tensor)
                    pred_class = torch.argmax(probs, dim=-1).item()
                    confidence = torch.max(probs).item()
                end_time = time.time()
                prediction_time = end_time - start_time
                print(f"Prediction Time: {prediction_time:.4f} seconds")

                # Display classification result
                label = "Good" if pred_class != NEG_CLASS else "Anomaly"
                print(f"Prediction: {label} (Confidence: {confidence:.2f})")
            else:
                print("⚠️ Region out of bounds!")

        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
