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
2. Place your image path in the `images` folder.
3. Update the code with the correct path of your `images.jpg`.
4. Run the `extract_center_area.py` script.

## Example

```python
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import cv2 #type: ignore

def extract_center_area(image, output_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur to reduce noise
    _, binary = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV) # Apply threshold to get binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Find contours    
    largest_contour = max(contours, key=cv2.contourArea) # Get the largest contour which will be the cut-out area
    mask = np.zeros_like(gray)  # Create a mask for the largest contour
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)    # Extract the center area using the mask 
    center_area = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(largest_contour) # Find bounding box coordinates to crop the center area
    cropped_center_area = center_area[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_center_area)
    
    return cropped_center_area
image = cv2.imread('path to ur image')
output_path = 'center_area1.png'
center_area1 = extract_center_area(image, output_path)
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image ')
plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(center_area1, cv2.COLOR_BGR2RGB)), plt.title('Center Area ')
plt.show()


Make sure:
1. The `center_area1.png` file is in the root of your repository if you reference it directly as `center_area1.png`.
2. If it’s in a folder like `images`, you need to update the path accordingly, such as `images/center_area1.png`.

### Uploading and Checking:

1. **Commit and Push**: Make sure you have committed the image file to your repository and pushed it to GitHub.
2. **Check File Location**: Verify the location of the image file on GitHub.

Here’s an example of how to add the image correctly if it’s in an `images` folder:

```markdown
![Center Area](images/center_area1.png)
