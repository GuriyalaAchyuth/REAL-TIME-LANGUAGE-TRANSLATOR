import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure

def load_image(image_path):
    """Load a grayscale medical image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or cannot be loaded.")
    return image

def preprocess_image(image):
    """Apply Gaussian blur and enhance contrast."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

def detect_calcium(image):
    """Detect calcium deposits based on intensity thresholding."""
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold
    return binary_image

def measure_calcium(binary_image):
    """Measure calcium regions in the binary image."""
    labeled_image = measure.label(binary_image)
    regions = measure.regionprops(labeled_image)
    calcium_areas = [region.area for region in regions]
    total_calcium = sum(calcium_areas)
    return total_calcium, calcium_areas

def main(image_path):
    """Main function to process and analyze calcium levels."""
    image = load_image(image_path)
    processed_image = preprocess_image(image)
    binary_image = detect_calcium(processed_image)
    total_calcium, calcium_areas = measure_calcium(binary_image)
    
    # Display results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(binary_image, cmap='gray')
    ax[1].set_title(f"Calcium Deposits (Total: {total_calcium} px)")
    ax[1].axis("off")
    
    plt.show()
    print(f"Total Calcium Deposit Area: {total_calcium} pixels")