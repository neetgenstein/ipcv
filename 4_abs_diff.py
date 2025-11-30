#pip install opencv-python


import cv2
import numpy as np

# Load two images (ensure both are same size and type)
M1 = cv2.imread('image1.png')   # Replace with your path
M2 = cv2.imread('image2.png')   # Replace with your path

# Resize to same size if necessary
M1 = cv2.resize(M1, (512, 512))
M2 = cv2.resize(M2, (512, 512))

# Compute absolute difference
Out = cv2.absdiff(M1, M2)

# Display results
cv2.imshow('Image 1 (M1)', M1)
cv2.imshow('Image 2 (M2)', M2)
cv2.imshow('Absolute Difference', Out)

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save output image
cv2.imwrite('difference_output.jpg', Out)