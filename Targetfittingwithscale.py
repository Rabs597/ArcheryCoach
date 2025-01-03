##Workign here.  Trying to get archery reference to play with the trackbars
# currently trying to implement a class for the test image but can't get it to load
# the image correctly.
import cv2
import numpy as np
import ArcheryReferenceTarget as ArcheryReferenceTarget
from skimage.metrics import structural_similarity as ssim

class ClickHandler:
    def __init__(self):
        self.clicked_coordinates = None

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_coordinates = (x, y)
            print(f"Clicked coordinates: {self.clicked_coordinates}")

class TestImage:
    def __init__(self, imgpath="input/image3.png"):
        self.original_image = self.loadimage(imgpath)
        self.overlaid_image = self.original_image.copy()
        self.centre_coordinates = np.array([0,0],dtype=int)

    def loadimage(self, imgpath):
        print("Loading image from path:", imgpath)  # Debug the file path
        tempimg = cv2.imread(imgpath)
        if tempimg is None:
            print("Error: Image not found at path:", imgpath)
        else:
            print("Image loaded successfully with shape:", tempimg.shape)
        return tempimg

def show_image_and_get_click(image, message):
    handler = ClickHandler()
    display_image = image.copy()
    cv2.putText(display_image, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Click to Get Coordinates", display_image)
    cv2.setMouseCallback("Click to Get Coordinates", handler.click_event)

    while handler.clicked_coordinates is None:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return handler.clicked_coordinates

def masked_ssim(image1, image2, mask):
    # Ensure the images and mask have the same size
    if image1.shape != image2.shape or image1.shape[:2] != mask.shape[:2]:
        raise ValueError("Images and mask must have the same dimensions")

    # Only keep the pixels where the mask is greater than zero
    masked_image1 = np.zeros_like(image1)
    masked_image2 = np.zeros_like(image2)

    # Apply the mask to the images (keep only the regions where mask is non-zero)
    masked_image1[mask > 0] = image1[mask > 0]
    masked_image2[mask > 0] = image2[mask > 0]

    # Now calculate SSIM for only the masked regions
    ssim_value = ssim(masked_image1, masked_image2, data_range=255, win_size=3, channel_axis=-1)
    
    return ssim_value

def update_overlay(testimage, dummytarget, dummyalphamask):
    result = testimage.copy()
    for c in range(3):
        result[:, :, c] = (dummytarget[:, :, c] * dummyalphamask + result[:, :, c] * (1 - dummyalphamask))
    return result

def on_trackbar_change(_):
    global scale, testImage, center_coordinates, color_radii

    try:
        scale = cv2.getTrackbarPos("Scale", "SSIM Visualization") / 1000.0
    except cv2.error:
        print("Error retrieving trackbar position.")
        return

    # Get perspective values from trackbars
    perspective_x = cv2.getTrackbarPos("Perspective X", "SSIM Visualization") / 100.0
    perspective_y = cv2.getTrackbarPos("Perspective Y", "SSIM Visualization") / 100.0
    
    # Create a perspective matrix based on the trackbar values
    perspective_matrix = np.array([[1, perspective_x, 0], [perspective_y, 1, 0], [0, 0, 1]])

    # Create the dummy target (without perspective shift applied yet)
    dummytarget, dummytargetmask = create_dummy_target(
        center_coordinates, color_radii, scale, testImage.original_image.shape
    )

    # Apply perspective transformation to the dummy target (before applying the mask)
    dummytarget = apply_perspective(dummytarget, perspective_matrix)
    dummytargetmask = apply_perspective(dummytargetmask, perspective_matrix)

    # Apply alpha blending to create the overlay
    dummyalphamask = 0.5 * cv2.cvtColor(dummytargetmask, cv2.COLOR_BGR2GRAY) / 255.0

    try:
        current_ssim = masked_ssim(testimage, dummytarget, dummytargetmask)
    except ValueError as e:
        print(f"SSIM Error: {e}")
        return

    result = update_overlay(testImage, dummytarget, dummyalphamask)
    cv2.putText(result, f"SSIM: {current_ssim:.4f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Resize the result to show 50% size
    result_resized = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("SSIM Visualization", result_resized)

def main():
    global referenceTarget, testImage

    # Load test image
    testimagepath = ("input/image2.png")
    testImage=TestImage(testimagepath)
    


    # User selects center and edge
    center_coordinates_in_test_image = show_image_and_get_click(testImage.original_image, "Click on the center")
    edge_coordinates_in_test_image = show_image_and_get_click(testImage.original_image, "Click the edge of the white")
    test_target_radius = np.linalg.norm(np.array(center_coordinates_in_test_image) - np.array(edge_coordinates_in_test_image))
    
    referenceTarget=ArcheryReferenceTarget(test_target_radius)

    # Trackbars for scale and perspective
    cv2.namedWindow("SSIM Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SSIM Visualization", 800, 600)  # Resize window to fit the trackbars
    cv2.createTrackbar("Scale", "SSIM Visualization", 1000, 2000, on_trackbar_change)  # 1.0 = 1000
    cv2.createTrackbar("Perspective X", "SSIM Visualization", 0, 100, on_trackbar_change)  # Perspective X
    cv2.createTrackbar("Perspective Y", "SSIM Visualization", 0, 100, on_trackbar_change)  # Perspective Y

    # Initial call to render the visualization
    on_trackbar_change(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
