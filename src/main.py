import cv2 as cv
from threshold import compute_histogram, compute_otsu_threshold, apply_threshold, dilate

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

# Show results
cv.imshow("Original Image", img)
cv.imshow("Binary Image (Otsu)", binary)
cv.imshow("Dilated Image", dilated)
cv.waitKey(0)
cv.destroyAllWindows()