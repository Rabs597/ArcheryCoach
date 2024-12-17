import cv2
import numpy as np
import numpy as np
from skimage.metrics import structural_similarity as ssim


refimage = cv2.imread('Target_refs/Fulltarget.jpg')
testimage = cv2.imread('input\image3.png')
refheight, refwidth = refimage.shape[:2]

# # Compute the center of the refimage
# refcenter_x, refcenter_y = refwidth / 2, refheight / 2

# # 1. Translation matrix to move the refimage center to the origin
# ref_center = np.array([
#     [1, 0, -refcenter_x],
#     [0, 1, -refcenter_y],
#     [0, 0, 1]
# ])

#### temp code to let the user select the centre, replace this with automatic based on colors
class ClickHandler:
    def __init__(self):
        self.clicked_coordinates = None  # Initialize the coordinates

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Store the coordinates as a member variable
            self.clicked_coordinates = (x, y)
            print(f"Clicked coordinates: {self.clicked_coordinates}")


# Function to show the image and record the click
def show_image_and_get_click(image,message):
    # Create an instance of the ClickHandler class
    handler = ClickHandler()

     # Display a message on the image
    display_image = image.copy()
    cv2.putText(display_image, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
     # Create a window and set the mouse callback
    cv2.imshow("Click to Get Coordinates", display_image)
    cv2.setMouseCallback("Click to Get Coordinates", handler.click_event)

    # Wait until the user clicks and coordinates are captured
    while handler.clicked_coordinates is None:
        cv2.waitKey(1)  # Refresh the OpenCV window

    # Close the window after a click
    cv2.destroyAllWindows()

    # Return the captured coordinates
    return handler.clicked_coordinates

def compute_ssim_weighted(image1, image2):
    # Load RGBA images
    image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    
    # Check dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Split channels
    r1, g1, b1, a1 = cv2.split(image1)
    r2, g2, b2, a2 = cv2.split(image2)

    # Normalize alpha to [0, 1]
    alpha_weight = a1 / 255.0

    # Weighted SSIM for RGB channels
    ssim_r = ssim(r1, r2, data_range=255) * alpha_weight.mean()
    ssim_g = ssim(g1, g2, data_range=255) * alpha_weight.mean()
    ssim_b = ssim(b1, b2, data_range=255) * alpha_weight.mean()
    
    # Average SSIM
    mean_ssim = (ssim_r + ssim_g + ssim_b) / 3.0
    return mean_ssim

def masked_ssim(image1, image2, mask):
    """
    Computes SSIM between two images with a mask.

    Parameters:
    - image1: First input image (grayscale or single-channel).
    - image2: Second input image (same size as image1).
    - mask: Binary mask (same size as the images). Non-zero values indicate regions to include.

    Returns:
    - SSIM value for the masked regions.
    """
    # Ensure the images and mask are the same size
    if image1.shape != image2.shape or image1.shape != mask.shape:
        raise ValueError("Images and mask must have the same dimensions")
    
    # Apply the mask to the images
    masked_image1 = image1 * (mask > 0)
    masked_image2 = image2 * (mask > 0)

    # Compute SSIM
    ssim_value = ssim(masked_image1, masked_image2, data_range=masked_image2.max() - masked_image2.min(),channel_axis=-1)
    return ssim_value

def main():
    # Get used to choose centre and an edge of red circle
    # Expect to need to make this more sophisticated and use ellipses, rather than just circles
    centre_coordinates = show_image_and_get_click(testimage.copy(),"Click on the center")
    blueedge_coordinates = show_image_and_get_click(testimage.copy(),"Click where red meets blue")
    #calculate color radii
    R=np.linalg.norm(np.array(centre_coordinates) - np.array(blueedge_coordinates))
    color_radii=np.array([2.5*R,2.0*R,1.5*R,1*R,0.5*R])

    ##generate dummy target on blank canvas to match original
    # Create a black (empty) canvas with the same size as the original image
    dummytarget= np.zeros((testimage.shape), dtype=np.uint8)
    dummytargetmask=np.zeros((testimage.shape), dtype=np.uint8)
    # Draw the circles (full target)
    cv2.circle(dummytarget, centre_coordinates, int(color_radii[0]), (255, 255, 255), -1)
    cv2.circle(dummytarget, centre_coordinates, int(color_radii[1]), (0, 0, 0), -1)
    cv2.circle(dummytarget, centre_coordinates, int(color_radii[2]), (255, 0, 0), -1)
    cv2.circle(dummytarget, centre_coordinates, int(color_radii[3]), (0, 0, 255), -1)
    cv2.circle(dummytarget, centre_coordinates, int(color_radii[4]), (0, 255, 255), -1)
    #create a mask for the target ROI
    cv2.circle(dummytargetmask, centre_coordinates, int(color_radii[0]), (255, 255, 255), -1)
    dummyalphamask=0.5*cv2.cvtColor(dummytargetmask, cv2.COLOR_BGR2GRAY)/ 255.0 
    ssim_value = masked_ssim(testimage, dummytarget, dummytargetmask)
    print(f"Masked SSIM: {ssim_value}")

    ##show calculated rings on original image
    #for radius in color_radii:
    #    # Draw the outline of the circle
    #    cv2.circle(testimage, centre_coordinates, int(radius), (0, 0, 255), 3)
    result = testimage.copy()
    for c in range(3):  # For each color channel (BGR)
        result[:, :, c] = (dummytarget[:, :, c] * dummyalphamask + result[:, :, c] * (1 - dummyalphamask))

    # Show the result
    cv2.imshow("Overlay with Transparency", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ##### NEXT: create a way of turning the target with sliders and showing the impact on SSIM to confirm it doe make a maximum
    ##### THEN: automate the varation of homography components

if __name__ == "__main__":
    main()