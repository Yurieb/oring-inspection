import cv2 as cv
import time
from threshold import compute_histogram, compute_otsu_threshold, apply_threshold, dilate, erode
from labeling import connected_components, component_areas, keep_largest_component, compute_perimeter, compute_circularity, find_interior_hole_area

img = cv.imread('images/Oring14.jpg', 0)

if img is None:
    print("Image failed to load.")
    exit()

print("Image shape:", img.shape)

# Start timer
t_start = time.perf_counter()

# Build histogram and find otsu threshold automatically
hist = compute_histogram(img)
threshold = compute_otsu_threshold(hist, img.size)
print("Otsu threshold value:", threshold)

# Apply threshold manually to get a black and white image
binary = apply_threshold(img, threshold)

# Use a lower threshold to help spot gaps in broken rings
binary_low = apply_threshold(img, threshold // 2)

# Find the hole inside the ring — a complete ring has a large hole and a broken one does not
hole_area = find_interior_hole_area(255 - binary_low)
print("Interior hole area:", hole_area)

# Morphology — close the small gaps and clean up noise
dilated = dilate(binary)
closed = erode(dilated)

# Invert ring becomes white
closed = 255 - closed

# Connected component labelling
labels, num_objects = connected_components(closed)
print("Number of connected components:", num_objects)

areas = component_areas(labels, num_objects)
print("Component areas:", areas)

clean_ring = keep_largest_component(labels, areas)
ring_area = max(areas)

perimeter = compute_perimeter(clean_ring)
print("Perimeter:", perimeter)

circularity = compute_circularity(ring_area, perimeter)
print("Circularity:", circularity)

# Stop timer
t_end = time.perf_counter()
processing_time = (t_end - t_start) * 1000
print(f"Processing time: {processing_time:.1f}ms")

# Decision logic
# Count meaningful components ignore tiny border artifacts under 2000 pixels
significant = sum(1 for a in areas if a > 2000)

# Classify the ring as pass or fail
if significant > 1:
    result_text = "FAIL - Extra object detected"
elif ring_area < 3000:
    result_text = "FAIL - Ring too small"
elif hole_area < 6000:
    result_text = "FAIL - Ring is open/broken"
elif hole_area > 15500:
    result_text = "FAIL - Ring is abnormal"
else:
    result_text = "PASS - Ring OK"

print("Result:", result_text)

# Convert to colour so we can draw coloured text
display = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Draw inspection result on image
cv.putText(display, result_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7,
           (0, 255, 0) if "PASS" in result_text else (0, 0, 255), 2)

# Draw processing time on image
cv.putText(display, f"Time: {processing_time:.1f}ms", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6,
           (255, 255, 255), 2)

# Show results
cv.imshow("Original Image", img)
cv.imshow("Binary Image (Otsu)", binary)
cv.imshow("Closed Image", closed)
cv.imshow("Largest Component Only", clean_ring)
cv.imshow("Inspection Result", display)

cv.waitKey(0)
cv.destroyAllWindows()