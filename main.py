import torch
import yolov5.models.yolo as yolo  # add this
torch.serialization.add_safe_globals([yolo.DetectionModel])
from VehicleDetectionProject.utils.helpers import assign_plate_to_vehicle, preprocess_plate_image, draw_text_with_background



import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import torch
from utils.helpers import assign_plate_to_vehicle, preprocess_plate_image, draw_text_with_background
from utils.tracker import VehicleTracker
import os


class VehiclePlateDetector:
    def __init__(self):
        # Initialize models

        print("Loading models...")
        self.vehicle_model = YOLO('yolov8s.pt')

        # Load license plate model
        try:
            self.plate_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                              'keremberke/yolov5m-license-plate',
                                              force_reload=False)
        except:
            print("Error loading plate model. Using backup method...")
            self.plate_model = YOLO('yolov8n.pt')  # Fallback

        # Initialize OCR
        print("Initializing OCR...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        # Initialize tracker
        self.tracker = VehicleTracker(max_disappeared=30, max_distance=100)

        # Vehicle classes in COCO dataset
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

        print("Initialization complete!")

    def detect_vehicles(self, frame):
        """Detect vehicles in frame"""
        results = self.vehicle_model(frame, verbose=False)
        vehicles = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.vehicle_model.names[class_id]
                    confidence = float(box.conf[0])

                    if class_name in self.vehicle_classes and confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicles.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class': class_name,
                            'centroid': ((x1 + x2) // 2, (y1 + y2) // 2)
                        })

        return vehicles

    def detect_license_plates(self, frame):
        """Detect license plates in frame"""
        try:
            results = self.plate_model(frame)
            plates = []

            # Handle different result formats
            if hasattr(results, 'xyxy'):
                # YOLOv5 format
                detections = results.xyxy[0].cpu().numpy()
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    if conf > 0.3:
                        plates.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf)
                        })
            else:
                # YOLOv8 format
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if float(box.conf[0]) > 0.3:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                plates.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(box.conf[0])
                                })

            return plates
        except Exception as e:
            print(f"Error in plate detection: {e}")
            return []

    def extract_plate_text(self, plate_img):
        """Extract text from license plate image"""
        try:
            # Preprocess image
            processed_img = preprocess_plate_image(plate_img)

            # Perform OCR
            results = self.ocr_reader.readtext(processed_img, detail=0)

            if results:
                # Combine all text results
                plate_text = ' '.join(results).strip()
                # Remove special characters and keep only alphanumeric
                plate_text = ''.join(c for c in plate_text if c.isalnum())
                return plate_text

            return ""
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def process_frame(self, frame, counting_line_y=None):
        """Process single frame"""
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)

        # Detect license plates
        plates = self.detect_license_plates(frame)

        # Update tracker
        if vehicles:
            centroids = [v['centroid'] for v in vehicles]
            tracked_objects = self.tracker.update(centroids, counting_line_y)
        else:
            tracked_objects = self.tracker.update([], counting_line_y)

        # Draw vehicles and assign plates
        for i, vehicle in enumerate(vehicles):
            x1, y1, x2, y2 = vehicle['bbox']

            # Draw vehicle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw vehicle info
            vehicle_text = f"{vehicle['class']}: {vehicle['confidence']:.2f}"
            draw_text_with_background(frame, vehicle_text, (x1, y1 - 10))

            # Find matching license plate
            plate_text = ""
            for plate in plates:
                px1, py1, px2, py2 = plate['bbox']

                # Check if plate is within vehicle bounds
                if assign_plate_to_vehicle([vehicle['bbox']], plate['bbox']) == 0:
                    # Draw plate bounding box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

                    # Extract plate region
                    plate_img = frame[py1:py2, px1:px2]
                    if plate_img.size > 0:
                        plate_text = self.extract_plate_text(plate_img)

                    break

            # Draw license plate text above vehicle
            if plate_text:
                draw_text_with_background(frame, f"Plate: {plate_text}",
                                          (x1, y1 - 40), font_scale=0.8,
                                          text_color=(0, 255, 255),
                                          background_color=(0, 0, 255))

        # Draw counting line
        if counting_line_y:
            cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y),
                     (0, 255, 255), 2)
            draw_text_with_background(frame, "COUNTING LINE", (10, counting_line_y - 10))

        # Draw vehicle count
        count_text = f"Vehicle Count: {self.tracker.vehicle_count}"
        draw_text_with_background(frame, count_text, (10, 30), font_scale=1.0,
                                  text_color=(255, 255, 255), background_color=(0, 0, 255))

        return frame

    def run_video(self, video_path, output_path=None):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set counting line (60% down the frame)
        counting_line_y = int(height * 0.6)

        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print("Processing video... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame, counting_line_y)

            # Write frame if output specified
            if out:
                out.write(processed_frame)

            # Display frame
            cv2.imshow('Vehicle and License Plate Detection', processed_frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames, Vehicle count: {self.tracker.vehicle_count}")

        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        print(f"Processing complete! Final vehicle count: {self.tracker.vehicle_count}")

    def run_webcam(self):
        """Run with webcam"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Get frame dimensions
        ret, frame = cap.read()
        if ret:
            height = frame.shape[0]
            counting_line_y = int(height * 0.6)

        print("Running with webcam... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame, counting_line_y)

            # Display frame
            cv2.imshow('Vehicle and License Plate Detection', processed_frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    # Initialize detector
    detector = VehiclePlateDetector()

    # Choose input method
    choice = input("Choose input method:\n1. Video file\n2. Webcam\nEnter choice (1/2): ")

    if choice == '1':
        video_path = input("Enter video file path (or press Enter for default): ").strip()
        if not video_path:
            video_path = "videos/sample_traffic.mp4"  # Default path

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            print("Please place a video file in the videos/ folder")
            return

        output_path = input("Enter output path (optional, press Enter to skip): ").strip()
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = None

        detector.run_video(video_path, output_path)

    elif choice == '2':
        detector.run_webcam()

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
