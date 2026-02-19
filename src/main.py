import cv2 as cv
import numpy as np

img = cv.imread('images/Oring1.jpg', 0)

print(img.shape)

cv.imshow("O-ring", img)
cv.waitKey(0)
cv.destroyAllWindows()