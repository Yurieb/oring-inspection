import numpy as np


def compute_histogram(img):
    # Create array for 256 grayscale levels 
    hist = np.zeros(256, dtype=int)

    # Go through every pixel in the image
    rows, cols = img.shape
    for x in range(rows):
        for y in range(cols):
            value = img[x, y]
            hist[value] += 1  

    return hist


def compute_otsu_threshold(hist, total_pixels):
    # Find threshold that best separates background and foreground
    best_threshold = 0
    max_variance = 0

    # Compute total intensity sum of image
    total_sum = 0
    for i in range(256):
        total_sum += i * hist[i]

    background_sum = 0
    background_weight = 0

    # Test every threshold possible from 0 to 255
    for t in range(256):
        background_weight += hist[t]

        if background_weight == 0:
            continue

        foreground_weight = total_pixels - background_weight
        if foreground_weight == 0:
            break

        background_sum += t * hist[t]

        mean_background = background_sum / background_weight
        mean_foreground = (total_sum - background_sum) / foreground_weight

        # Between class variance formula
        variance = background_weight * foreground_weight * (mean_background - mean_foreground) ** 2

        # Keep the threshold that gives the highest variance
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return best_threshold


def apply_threshold(img, threshold):
    # Create empty binary image
    rows, cols = img.shape
    binary = np.zeros((rows, cols), dtype=np.uint8)

    # Convert image to black and white using threshold
    for x in range(rows):
        for y in range(cols):
            if img[x, y] > threshold:
                binary[x, y] = 255 
            else:
                binary[x, y] = 0    

    return binary