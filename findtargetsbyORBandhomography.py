##This doesn't work becuase it can't find any corners!

import cv2
import numpy as np

# Load the source and target images
image1 = cv2.imread('Target_refs\Fulltarget.jpg')
image2 = cv2.imread('input\image2.png')
height = 500
image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Create a Brute Force Matcher and match the descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches based on distance (lower distance means better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the first 10 matches (optional)
img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)

# Extract the matching points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the homography matrix using RANSAC
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# Use the homography matrix to align (warp) image1 onto image2
height, width, channels = image2.shape
aligned_image = cv2.warpPerspective(image1, H, (width, height))

# Display the result
cv2.imshow('Aligned Image', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()