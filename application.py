import tkinter as tk
from tkinter import Canvas, Button, Toplevel, Label, filedialog
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms

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

class HandwritingApp:
    def __init__(self, master):
        self.master = master
        master.title("SD19 Handwriting Recognition")

        # Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load PyTorch model
        self.model = SD19Model(num_classes=47).to(self.device)
        self.model.load_state_dict(torch.load('sd19_model.pth', map_location=self.device))
        self.model.eval()

        # Create drawing canvas
        self.canvas = Canvas(master, width=400, height=400, bg='black')
        self.canvas.pack()

        # Create buttons
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_recognize = Button(self.btn_frame, text="Recognize", command=self.predict)
        self.btn_recognize.pack(side=tk.LEFT, padx=10)

        self.btn_clear = Button(self.btn_frame, text="Clear", command=self.clear)
        self.btn_clear.pack(side=tk.RIGHT, padx=10)

        self.btn_upload = Button(self.btn_frame, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(side=tk.LEFT, padx=10)

        # Initialize image for drawing
        self.image = Image.new('L', (400, 400), 'black')
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

        # Class mapping
        self.class_mapping = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C.c', 'D', 'E', 'F', 'G', 'H', 'I.i', 'J.j',
            'K.k', 'L.l', 'M.m', 'N', 'O.o', 'P.p', 'Q', 'R', 'S.s',
            'T', 'U.u', 'V.v', 'W.w', 'X.x', 'Y.y', 'Z.z',
            'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
        ]
    
    def paint(self, event):
        # Draw on canvas and image
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill='white')

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (400, 400), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Preprocess image
        img = self.image.resize((128, 128))
        img = ImageOps.invert(img)
        img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)  # SD19 orientation fix

        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize if needed
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

        # Show result
        result_window = Toplevel(self.master)
        result_window.title("Prediction Result")
        label_text = f"Character: {self.class_mapping[predicted_class]}\nConfidence: {confidence * 100:.2f}%"
        Label(result_window, text=label_text, font=('Arial', 16)).pack(padx=20, pady=20)

    def upload_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            # Open and process the selected image
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img = img.resize((128, 128))  # Resize to 128x128 as per model input
            img = ImageOps.invert(img)  # Invert colors (white on black background)
            img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)  # Correct orientation

            # Convert image to tensor for model input
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Normalize if needed
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

            # Show result in a new window
            result_window = Toplevel(self.master)
            result_window.title("Prediction Result")
            label_text = f"Character: {self.class_mapping[predicted_class]}\nConfidence: {confidence * 100:.2f}%"
            Label(result_window, text=label_text, font=('Arial', 16)).pack(padx=20, pady=20)

# Initialize Tkinter window
root = tk.Tk()
app = HandwritingApp(root)
root.mainloop()
