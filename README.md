# Center Area Extraction

This repository contains the code to extract the center area (cut-out area) from provided images using OpenCV.

## Steps
1. **Read the Image**: Load the image using OpenCV.
2. **Convert to Grayscale**: Convert the image to grayscale for easier processing.
3. **Thresholding**: Apply a threshold to create a binary image.
4. **Find Contours**: Detect the contours in the binary image.
5. **Extract Region of Interest (ROI)**: Identify and extract the ROI based on the contours.
6. **Save the Extracted ROI**: Save the extracted center area as a new image.

## Usage
1. Clone the repository.
2. Place your images in the `images` folder.
3. Run the `extract_center_area.py` script.

## Example

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_center_area(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    center_area = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_center_area = center_area[y:y+h, x:x+w]
    return cropped_center_area

image_path = 'images/your_image.png'
center_area = extract_center_area(image_path)
cv2.imwrite('output/center_area.png', center_area)
