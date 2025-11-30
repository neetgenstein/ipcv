
#pip install opencv-python

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1) Preprocess the image (convert to grayscale + resize to 64x128)
# ---------------------------------------------------
img = cv2.imread("dog1.jpg", cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (64, 128))

# ---------------------------------------------------
# 2) Compute Gradients using Sobel (Gx, Gy)
# ---------------------------------------------------
Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# ---------------------------------------------------
# 3) Magnitude and Orientation
# ---------------------------------------------------
magnitude = np.sqrt(Gx**2 + Gy**2)
orientation = np.rad2deg(np.arctan2(Gy, Gx)) % 180  # HOG uses unsigned angles [0,180)

# ---------------------------------------------------
# 4) Histogram of Gradients in 8×8 cells (9 bins)
# ---------------------------------------------------
cell_size = 8
num_bins = 9
bin_size = 180 // num_bins

cells_x = img.shape[1] // cell_size
cells_y = img.shape[0] // cell_size

hog_cells = np.zeros((cells_y, cells_x, num_bins))

for i in range(cells_y):
    for j in range(cells_x):
        # Extract 8x8 cell
        mag_cell = magnitude[i*cell_size:(i+1)*cell_size,
                             j*cell_size:(j+1)*cell_size]
        ori_cell = orientation[i*cell_size:(i+1)*cell_size,
                               j*cell_size:(j+1)*cell_size]
        
        # Build histogram
        hist = np.zeros(num_bins)
        for y in range(cell_size):
            for x in range(cell_size):
                mag = mag_cell[y, x]
                angle = ori_cell[y, x]

                bin_idx = int(angle // bin_size) % num_bins
                hist[bin_idx] += mag
        
        hog_cells[i, j] = hist

# ---------------------------------------------------
# 5) Normalize using 16×16 blocks (2x2 cells)
# ---------------------------------------------------
block_size = 2   # (2 cells × 8px = 16px)
eps = 1e-5
hog_features = []

for i in range(cells_y - 1):
    for j in range(cells_x - 1):
        # 2x2 block containing 4 histograms
        block = hog_cells[i:i+block_size, j:j+block_size].reshape(-1)
        
        # L2 normalization
        norm_block = block / np.sqrt(np.sum(block**2) + eps)
        hog_features.extend(norm_block)

hog_features = np.array(hog_features)

# ---------------------------------------------------
# 6) Final HOG Feature Vector
# ---------------------------------------------------
print("HOG Feature Vector Length:", len(hog_features))
print("HOG Feature Vector:\n", hog_features)
