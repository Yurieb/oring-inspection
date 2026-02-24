import cv2 as cv
from threshold import compute_histogram, compute_otsu_threshold, apply_threshold, dilate, erode
from labeling import connected_components, component_areas

img = cv.imread('images/Oring1.jpg', 0)

if img is None:
    print("Image failed to load.")
    exit()

print("Image shape:", img.shape)


hist = compute_histogram(img)

# Compute automatic threshold using Otsu
threshold = compute_otsu_threshold(hist, img.size)
print("Otsu threshold value:", threshold)

# Apply threshold manually
binary = apply_threshold(img, threshold)

# Morphology
dilated = dilate(binary)
closed = erode(dilated)

# Invert ring becomes white
closed = 255 - closed

labels, num_objects = connected_components(closed)
print("Number of connected components:", num_objects)

areas = component_areas(labels, num_objects)
print("Component areas:", areas)

# Show results
cv.imshow("Original Image", img)
cv.imshow("Binary Image (Otsu)", binary)
cv.imshow("Dilated Image", dilated)
cv.imshow("Closed Image", closed)
cv.imshow("Closed Image", closed)

cv.waitKey(0)
cv.destroyAllWindows()