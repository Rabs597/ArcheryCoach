

import cv2
import numpy as np


refimage = cv2.imread('Target_refs/Fulltarget.jpg')
testimage = cv2.imread('input\image2.png')

#FIND the centre of the target
#Option A: find the gold, then fit from there.
    # tried finding the yellow circles and it got confused by other yellow patches, 
    # which means it won't be robust to lighting
    #could try the red, as this is less common?
#OptionB: could try finding multiple circles and taking the centre of them?
#OptionC: full sliding of homography over the image - brute force

#Implementing Option A
#OLD CODE FOR OPTION A of finding the centre of the target from findtarget.py
#scan an image for a target by checking for the yellow centre
hsv = cv2.cvtColor(testimage, cv2.COLOR_BGR2HSV)

# Define the range for yellow color in HSV space
# Lower and upper bounds for yellow color (you may need to adjust these values)
lower_color = np.array([160, 100, 100])  # Lower bound for red
upper_color = np.array([200, 255, 255]) # Upper bound for red (first range of red hue)

# Create a mask to extract the red parts of the image
mask = cv2.inRange(hsv, lower_color, upper_color)

# Perform morphological operations (optional, but helps clean noise)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and find the largest circle (assumed yellow circle)
for contour in contours:
    # Fit a circle to the contour using minEnclosingCircle
    (x, y), radius = cv2.minEnclosingCircle(contour)
    
    # You can filter out small contours based on the radius size if needed
    if radius > 50:  # Arbitrary threshold to avoid noise
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw the circle on the image
        cv2.circle(testimage, center, radius, (0, 255, 0), 2)  # Draw the circle (green)
        cv2.circle(testimage, center, 5, (0, 0, 255), -1)  # Mark the center (red)

# Display the result
color_regions = cv2.bitwise_and(testimage, testimage, mask=mask)


height = 500 
testimage_resized = cv2.resize(testimage, (int(testimage.shape[1] * height / testimage.shape[0]), height))
refimage_resized = cv2.resize(refimage, (int(refimage.shape[1] * height / refimage.shape[0]), height))
outputimage_resized = cv2.resize(testimage, (int(testimage.shape[1] * height / testimage.shape[0]), height))

combined_image = np.hstack((outputimage_resized, outputimage_resized))
cv2.imshow('fulltarget', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




