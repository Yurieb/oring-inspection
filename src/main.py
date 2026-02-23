import cv2 as cv
from threshold import compute_histogram, compute_otsu_threshold, apply_threshold, dilate, erode

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

dilated = dilate(binary)

eroded = erode(binary)

closed = erode(dilated)

# Show results
cv.imshow("Original Image", img)
cv.imshow("Binary Image (Otsu)", binary)
cv.imshow("Dilated Image", dilated)
cv.imshow("Eroded Image", eroded)
cv.imshow("Closed Image", closed)

cv.waitKey(0)
cv.destroyAllWindows()