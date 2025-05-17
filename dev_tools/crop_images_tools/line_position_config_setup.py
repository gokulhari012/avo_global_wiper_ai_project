import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import json

working_dir_path = r"C:\Users\gokul\Documents\projects\avo global wiper\global_wiper_final\\"
# Read path from file
# with open("global_wiper_final/working_dir.txt", "r") as f:
with open("working_dir.txt", "r") as f:
    working_dir_path = f.read().strip()
line_positions_json_file = working_dir_path+"line_positions.json"
cropped_image_path = working_dir_path+"cropped_images"

class LineDrawer:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Crop Tool")
        self.image_path = image_path

        # Load original image
        self.original_image = Image.open(image_path)
        self.ori_width, self.ori_height = self.original_image.size

        # Display resolution (scaled down for smoother interaction)
        self.display_width = 1280
        self.scale_ratio = self.display_width / self.ori_width
        self.display_height = int(self.ori_height * self.scale_ratio)

        self.display_image = self.original_image.resize(
            (self.display_width, self.display_height), Image.Resampling.LANCZOS
        )

        # Tkinter layout
        outer_frame = tk.Frame(self.root)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(outer_frame, width=800, height=600, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add scrollbars
        h_scroll = tk.Scrollbar(outer_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        v_scroll = tk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.config(scrollregion=(0, 0, self.display_width, self.display_height))

        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Create 2 vertical and 11 horizontal lines
        self.v_lines = [
            self.canvas.create_line(x, 0, x, self.display_height, fill="red", width=2)
            for x in [100, 300]
        ]
        self.h_lines = [
            self.canvas.create_line(0, y, self.display_width, y, fill="blue", width=2)
            for y in range(0, self.display_height, self.display_height // 11)
        ]

        self.drag_data = {"line": None, "type": None}
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        # Save Button
        self.save_button = tk.Button(self.root, text="💾 Save Crops", command=self.save_crops)
        self.save_button.pack(pady=10)
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
        # Get positions on display canvas
        # x_disp = sorted([self.canvas.coords(line)[0] for line in self.v_lines])
        # y_disp = sorted([self.canvas.coords(line)[1] for line in self.h_lines])

        # # Scale to original image resolution
        # x_positions = [int(x / self.scale_ratio) for x in x_disp]
        # y_positions = [int(y / self.scale_ratio) for y in y_disp]

        # image_cv = cv2.imread(self.image_path)
        # folder = cropped_image_path
        # os.makedirs(folder, exist_ok=True)
        # print(y_positions)datetime
        # for i in range(10):
        #     y1 = y_positions[i+1]
        #     y2 = y_positions[i+1 + 1]
        #     x1 = x_positions[0]
        #     x2 = x_positions[1]
        #     crop = image_cv[y1:y2, x1:x2]
            #cv2.imwrite(f"{folder}/crop_{i+1}.png", crop)
        self.save_line_positions_to_json()
        print(f"✅ Saved 10 line position in json file")

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

def main():

    # image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    image_path = working_dir_path + "captured_images/image_None.jpg"
    if not image_path:
        return

    root = tk.Tk()
    app = LineDrawer(root, image_path)
    root.mainloop()

if __name__ == "__main__":
    main()
