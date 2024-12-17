##this version does an ok job of getting the target to the centre and applying perspective shifts to the target (although they are not centred correctly)
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class ClickHandler:
    def __init__(self):
        self.clicked_coordinates = None

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_coordinates = (x, y)
            print(f"Clicked coordinates: {self.clicked_coordinates}")

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

def create_normalised_target(size):
    normalised_target = np.zeros(padded_shape = (size, size, shape[2]), dtype=np.uint8)
    
    return normalised_target

def create_dummy_target(center, radii, scale, shape, padding=500):
    # Expand the canvas to prevent cropping during scaling
    padded_shape = (shape[0] + 2 * padding, shape[1] + 2 * padding, shape[2])
    canvas = np.zeros(padded_shape, dtype=np.uint8)
    mask = np.zeros(padded_shape, dtype=np.uint8)
    
    # Adjust the center to account for the padding
    padded_center = (center[0] + padding, center[1] + padding)

    # Create the scaling matrix (centering on the circle center)
    translation_to_origin = np.array([[1, 0, -padded_center[0]], [0, 1, -padded_center[1]], [0, 0, 1]])
    scaling_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])  # Use single scale for both x and y
    translation_back = np.array([[1, 0, padded_center[0]], [0, 1, padded_center[1]], [0, 0, 1]])

    combined_matrix = translation_back @ scaling_matrix @ translation_to_origin
    affine_matrix = combined_matrix[:2, :]  # Extract 2x3 affine matrix for OpenCV

    # Draw the circles on the expanded canvas
    for i, radius in enumerate(radii):
        color = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)][i]
        cv2.circle(canvas, padded_center, int(radius), color, -1)
        cv2.circle(mask, padded_center, int(radius), (255, 255, 255), -1)

    # Apply scaling transformation
    transformed_canvas = cv2.warpAffine(canvas, affine_matrix, (padded_shape[1], padded_shape[0]))
    transformed_mask = cv2.warpAffine(mask, affine_matrix, (padded_shape[1], padded_shape[0]))

    # Crop the canvas back to the original image size
    cropped_canvas = transformed_canvas[padding:padding + shape[0], padding:padding + shape[1]]
    cropped_mask = transformed_mask[padding:padding + shape[0], padding:padding + shape[1]]
    
    return cropped_canvas, cropped_mask

def apply_perspective(dummytarget, perspective_matrix):
    # Apply the homography transformation (perspective)
    transformed_target = cv2.warpPerspective(dummytarget, perspective_matrix, (dummytarget.shape[1], dummytarget.shape[0]))
    return transformed_target

def update_overlay(testimage, dummytarget, dummyalphamask):
    result = testimage.copy()
    for c in range(3):
        result[:, :, c] = (dummytarget[:, :, c] * dummyalphamask + result[:, :, c] * (1 - dummyalphamask))
    return result

def on_trackbar_change(_):
    global scale, testimage, center_coordinates, color_radii

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
        center_coordinates, color_radii, scale, testimage.shape
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

    result = update_overlay(testimage, dummytarget, dummyalphamask)
    cv2.putText(result, f"SSIM: {current_ssim:.4f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Resize the result to show 50% size
    result_resized = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("SSIM Visualization", result_resized)

def main():
    global testimage, center_coordinates, color_radii, scale

    # Load test image
    testimage = cv2.imread(r"input\\image3.png")
    if testimage is None:
        raise FileNotFoundError("Test image not found.")

    # User selects center and edge
    center_coordinates = show_image_and_get_click(testimage.copy(), "Click on the center")
    edge_coordinates = show_image_and_get_click(testimage.copy(), "Click where red meets blue")

    R = np.linalg.norm(np.array(center_coordinates) - np.array(edge_coordinates))
    color_radii = np.array([2.5 * R, 2.0 * R, 1.5 * R, 1 * R, 0.5 * R])

    # Initial scaling factor
    scale = 1.0

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
