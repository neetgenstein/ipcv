#pip install opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) Load image (grayscale)
# ---------------------------------------------------------
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------
# 2) Create SIFT object
#    (This internally handles:
#     - Scale space construction
#     - Keypoint localization
#     - Orientation assignment
#     - Descriptor computation)
# ---------------------------------------------------------
sift = cv2.SIFT_create()

# ---------------------------------------------------------
# 3) Detect keypoints and compute descriptors
# ---------------------------------------------------------
keypoints, descriptors = sift.detectAndCompute(img, None)

print("Number of keypoints detected:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

# ---------------------------------------------------------
# 4) Draw keypoints on the image
# ---------------------------------------------------------
output_img = cv2.drawKeypoints(
    img, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# ---------------------------------------------------------
# 5) Show result
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(output_img, cmap='gray')
plt.title("SIFT Keypoints with Orientation + Scale")
plt.axis("off")
plt.show()

# Optional: print descriptor details
print("Descriptor shape:", descriptors.shape)
print("Each descriptor length:", len(descriptors[0]))