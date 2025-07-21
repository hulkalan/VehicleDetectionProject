import numpy as np
import cv2


def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def assign_plate_to_vehicle(vehicle_boxes, plate_box, threshold=0.3):
    """Assign a license plate to the best matching vehicle"""
    best_iou = 0
    best_vehicle_idx = -1

    for i, vehicle_box in enumerate(vehicle_boxes):
        iou = calculate_iou(vehicle_box, plate_box)
        if iou > best_iou and iou > threshold:
            best_iou = iou
            best_vehicle_idx = i

    return best_vehicle_idx


def preprocess_plate_image(plate_img):
    """Preprocess license plate image for better OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return thresh


def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=0.6, text_color=(255, 255, 255),
                              background_color=(0, 0, 0), thickness=2):
    """Draw text with background for better visibility"""
    x, y = position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Draw background rectangle
    cv2.rectangle(img, (x - 5, y - text_size[1] - 10),
                  (x + text_size[0] + 5, y + 5), background_color, -1)

    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
