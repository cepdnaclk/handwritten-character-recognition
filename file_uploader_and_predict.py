import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import tkinter as tk
from tkinter import filedialog, Canvas
import numpy as np

# Correct label mapping for by_merge dataset
class_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C.c', 'D', 'E', 'F', 'G', 'H', 'I.i', 'J.j',
    'K.k', 'L.l', 'M.m', 'N', 'O.o', 'P.p', 'Q', 'R', 'S.s',
    'T', 'U.u', 'V.v', 'W.w', 'X.x', 'Y.y', 'Z.z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# Define the model (must match training architecture)
class SD19Model(nn.Module):
    def __init__(self, num_classes=47):
        super(SD19Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self._init_linear()
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _init_linear(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)
            features = self.features(dummy)
            self.flattened_size = features.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the trained model
model = SD19Model(num_classes=47)
model.load_state_dict(torch.load('sd19_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image):
    """Predict the character from a PIL Image."""
    image = transform(image).unsqueeze(0)  # Convert to tensor and add batch dimension
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_labels[predicted.item()]

# GUI Application
class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Character Recognition")

        # Upload Image Button
        self.upload_btn = tk.Button(root, text="Upload Image & Predict", command=self.upload_and_predict)
        self.upload_btn.pack(pady=10)

        # Drawing Canvas
        self.canvas = Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons for Prediction & Clearing Canvas
        self.predict_btn = tk.Button(root, text="Predict Drawn Character", command=self.predict_drawn_character)
        self.predict_btn.pack(pady=5)

        self.clear_btn = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(pady=5)

        # Label for displaying the prediction result
        self.result_label = tk.Label(root, text="Predicted Character: ", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Create an empty image for drawing
        self.image = Image.new("L", (300, 300), 255)
        self.draw = ImageOps.invert(self.image).convert("RGB")

    def upload_and_predict(self):
        """Handles file upload and prediction display."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            image = Image.open(file_path).convert("L")  # Convert to grayscale
            prediction = predict_image(image)
            self.result_label.config(text=f"Predicted Character: {prediction}")

    def paint(self, event):
        """Draw on the canvas using the mouse."""
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill="black", outline="black")

        # Draw on the image
        draw_img = ImageOps.invert(self.image).convert("RGB")
        draw_img = np.array(draw_img)
        draw_img[y:y+10, x:x+10] = 0  # Draw in black
        self.image = Image.fromarray(draw_img).convert("L")

    def predict_drawn_character(self):
        """Predicts the character from the drawn image on the canvas."""
        # Resize and preprocess drawn image
        image = self.image.resize((128, 128))
        prediction = predict_image(image)
        self.result_label.config(text=f"Predicted Character: {prediction}")

    def clear_canvas(self):
        """Clears the drawing canvas."""
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), 255)

# Run the Tkinter App
root = tk.Tk()
app = HandwritingApp(root)
root.mainloop()
