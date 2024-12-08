import cv2

imageA = cv2.imread('Target_refs/Fulltarget.jpg')
imageB = cv2.imread('Target_refs/6ringtarget.jpeg')

bullseye_point = (325,309)
inner_diameter_px = 50
inner_diameter_inch = 1.5
rings_amount = 6


cv2.imshow('fulltarget', imageA)
cv2.imshow('6ringtarget', imageB)
cv2.waitKey(0)
cv2.destroyAllWindows()