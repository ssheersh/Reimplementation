import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread('images/Rubber_Soul.jpg')
gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
im2 = cv.imread('images/All_Albums.jpg')
gray_2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT.create()
kp1, desc1 = sift.detectAndCompute(gray, None)
kp2, desc2 = sift.detectAndCompute(gray_2, None)

im1_k = cv.drawKeypoints(im1.copy(), kp1, im1.copy())
cv.imshow('sift_keypoints', im1_k)
im2_k = cv.drawKeypoints(im2.copy(), kp2, im2.copy())
cv.imshow('Sift_kp_2', im2_k)


# Use FLANN-based matcher to match the descriptors
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)

# Store only good matches using Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract the matched keypoints' locations
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute the homography matrix using RANSAC
H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)

# Get the size of the original image (im1) to ensure correct dimensions
height, width, channels = im1.shape

# Apply the inverse warp using the homography matrix
unwarped_image = cv.warpPerspective(im2, H, (width, height))

# Save or display the result
cv.imwrite('unwarped_image.jpg', unwarped_image)
plt.imshow(cv.cvtColor(unwarped_image, cv.COLOR_BGR2RGB))
plt.title('Unwarped Image')
plt.show()
cv.waitKey(0)
