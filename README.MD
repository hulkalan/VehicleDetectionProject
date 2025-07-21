# Vehicle and License Plate Detection System

**Real-time vehicle counting, tracking, and license-plate recognition** using YOLOv8, YOLOv5, and EasyOCR.

## 🚀 Features

- Detects and tracks vehicles (car, truck, bus, motorcycle) in video or webcam stream  
- Counts vehicles crossing a configurable reference line  
- Localizes license plates within each vehicle ROI  
- Performs OCR on detected plates and overlays plate numbers above vehicles  
- Saves processed video output (optional)  
- Supports both video-file and webcam inputs  

## 📂 Repository Structure

```
VehicleDetectionProject/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── config.py
├── download_models.py
├── main.py
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── tracker.py
├── models/                # (Ignored by Git) downloaded model weights
├── videos/                # sample input videos (small files only)
└── output/                # processed video outputs
```

## ⚙️ Prerequisites

- Python 3.8 or higher  
- (Optional) CUDA-capable GPU for accelerated inference  

## 🛠️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/VehicleDetectionProject.git
   cd VehicleDetectionProject
   ```

2. **Create a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate.bat       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**  
   ```bash
   python download_models.py
   ```

## 📝 Configuration

Edit `config.py` to adjust:

```python
# Model settings
VEHICLE_MODEL = 'yolov8s.pt'
PLATE_MODEL   = 'keremberke/yolov5m-license-plate'
VEHICLE_CONF  = 0.5
PLATE_CONF    = 0.3

# Tracking settings
MAX_DISAPPEARED = 30
MAX_DISTANCE    = 100

# Video settings
COUNT_LINE_RATIO = 0.6  # 60% down the frame
```

## ▶️ Usage

### 1. Process a Video File

```bash
python main.py
```

- Choose **“1”** when prompted  
- Enter path to input video (e.g., `videos/sample_traffic.mp4`)  
- (Optional) Specify an output path to save annotated video (e.g., `output/result.mp4`)  

### 2. Real-Time Webcam

```bash
python main.py
```

- Choose **“2”** when prompted  
- Stream runs until you press **q**  

## 🛡️ .gitignore

Ensure the following (among others) are ignored:

```
# Virtual envs
venv/
.env

# Models & large files
models/
*.pt

# Videos
videos/
output/

# Python cache
__pycache__/
```

## 🧪 Testing

> *Coming soon:* unit tests for helper functions in `utils/`.

## 📈 Performance Tips

- Use smaller YOLO models (e.g., `yolov8n.pt`) for faster FPS  
- Enable GPU inference: verify `torch.cuda.is_available()`  
- Reduce frame resolution or skip frames to lighten workload  

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com)  
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)  
- [EasyOCR Project](https://github.com/JaidedAI/EasyOCR)  
- [Roboflow Vehicle Detection Models](https://roboflow.com)  

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.