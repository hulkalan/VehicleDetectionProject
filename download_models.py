from yolov5 import load  # pip install yolov5


import torch
from ultralytics import YOLO
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Download vehicle detection model (YOLOv8)
print("Downloading vehicle detection model...")
vehicle_model = YOLO('yolov8s.pt')  # This auto-downloads
print("Vehicle model downloaded!")

# Download license plate detection model
print("Downloading license plate detection model...")
# Method 1: Using Hugging Face model
plate_model = load('keremberke/yolov5m-license-plate')
torch.save(plate_model.model.state_dict(), 'models/license_plate_model.pt')
print("License plate model downloaded!")

print("All models downloaded successfully!")
