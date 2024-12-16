
import cv2
import numpy as np

def find_circles(image, min_radius=10, max_circles=5):
    # Extract the red channel
    red_channel = image[:, :, 2]

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(red_channel, (15, 15), 0)

    # Detect circles using HoughCircles
    # The parameters can be adjusted based on the image
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT,  # Detection method
        dp=1,  # The inverse ratio of resolution
        minDist=1,  # Minimum distance between centers of detected circles
        param1=100,  # Higher threshold for edge detection (can be adjusted)
        param2=80,  # Accumulator threshold for center detection
        minRadius=min_radius,  # Minimum radius of circles to detect
        maxRadius=0  # Maximum radius of circles to detect
    )

    # If some circles are detected, they are returned as an array
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Convert to integer values
        # This assumes that circles[0] is a list of detected circles
        circles_sorted = sorted(circles[0], key=lambda x: x[2], reverse=True)

        # Limit the number of circles
        circles_sorted = circles_sorted[:max_circles]
        
        # Draw the detected circles on the original image
        for circle in circles_sorted:
            center = (circle[0], circle[1])  # (x, y) center of the circle
            radius = circle[2]  # Radius of the circle
            
            # Draw the center of the circle
            cv2.circle(image, center, 2, (0, 255, 0), 3)
            # Draw the outline of the circle
            cv2.circle(image, center, radius, (0, 0, 255), 3)
    
    else:
        # If no circles were detected
        print("No circles were found in the image.")

    return image, circles  # Return the image with circles drawn and the circles list


def find_ellipses(image, max_ellipses=5):
    # Extract the red channel
    red_channel = image[:, :, 2]

    # Apply GaussianBlur to reduce noise and improve ellipse detection
    blurred = cv2.GaussianBlur(red_channel, (15, 15), 0)

    # Use edge detection to find contours (Canny edge detection)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []

    for contour in contours:
        if len(contour) >= 5:  # At least 5 points are required to fit an ellipse
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)

            # Append the ellipse parameters (center, axes, angle)
            ellipses.append(ellipse)

    if len(ellipses) > 0:
        # Sort ellipses based on the area (major axis * minor axis), largest first
        ellipses_sorted = sorted(ellipses, key=lambda x: x[1][0] * x[1][1], reverse=True)

        # Limit the number of ellipses
        ellipses_sorted = ellipses_sorted[:max_ellipses]

        # Draw the ellipses on the image
        for ellipse in ellipses_sorted:
            center, axes, angle = ellipse
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Draw the ellipse
            cv2.circle(image, (int(center[0]), int(center[1])), 2, (0, 255, 0), 3)  # Draw the center

    else:
        # If no ellipses were detected
        print("No ellipses were found in the image.")

    return image


###############################################

refimage = cv2.imread('Target_refs/Fulltarget.jpg')
testimage = cv2.imread('input\image2.png')

refheight, refwidth = refimage.shape[:2]

# Compute the center of the refimage
refcenter_x, refcenter_y = refwidth / 2, refheight / 2

# 1. Translation matrix to move the refimage center to the origin
ref_center = np.array([
    [1, 0, -refcenter_x],
    [0, 1, -refcenter_y],
    [0, 0, 1]
])

circles_image=testimage.copy()
ellipse_image=testimage.copy()
circles_image, circles = find_circles(circles_image,int(circles_image.shape[1]/20),10)
ellipse_image, ellipses = find_ellipses(ellipse_image,10)
height = 500 
testimage_resized = cv2.resize(testimage, (int(testimage.shape[1] * height / testimage.shape[0]), height))
refimage_resized = cv2.resize(refimage, (int(refimage.shape[1] * height / refimage.shape[0]), height))
outputimage_resized = cv2.resize(output_image, (int(output_image.shape[1] * height / output_image.shape[0]), height))

combined_image = np.hstack((outputimage_resized, outputimage_resized))
cv2.imshow('fulltarget', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#FIND the centre of the target
#Option A: find the gold, then fit from there.
    # tried finding the yellow circles and it got confused by other yellow patches, 
    # which means it won't be robust to lighting
    #could try the red, as this is less common?
#OptionB: could try finding multiple circles and taking the centre of them?
#OptionC: full sliding of homography over the image - brute force

#Implementing Option B


#identify the centre
# test fit a reference target
# vary homography parameters to find best fit


# # Rotation angle in degrees
# theta = 20  # 45 degrees
# theta_rad = np.radians(theta)  # Convert to radians

# warpedimagesize=imageA.shape

# bullseye_point_in_ref_image= (325,309)
# inner_diameter_px = 50
# inner_diameter_inch = 1.5
# rings_amount = 6

# testtargetsize=0.5
# testcentredisplacement=[0,0]


# height, width = imageA.shape[:2]


# # Rotation angle in degrees
# theta = 20  # 45 degrees
# theta_rad = np.radians(theta)  # Convert to radians


# # 2. Rotation matrix for rotating around the origin
# R = np.array([
#     [np.cos(theta_rad), -np.sin(theta_rad), 0],
#     [np.sin(theta_rad), np.cos(theta_rad), 0],
#     [0, 0, 1]
# ])

# # 3. Inverse translation matrix to move the image back
# T_center_inv = np.array([
#     [1, 0, center_x],
#     [0, 1, center_y],
#     [0, 0, 1]
# ])

# # 4. Final homography matrix
# M = np.dot(T_center_inv, np.dot(R, T_center))


# warped=cv2.warpPerspective(imageA, M, warpedimagesize[:2])
# combined_image = np.hstack((imageA, warped))
# cv2.imshow('fulltarget', combined_image)
# #cv2.imshow('warped', warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()