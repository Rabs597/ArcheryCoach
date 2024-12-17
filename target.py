import numpy as np
import cv2

class ArcheryTarget:
    def __init__(self, R):
        self.original_radius = R  # Store the original radius
        self.centre = np.array([R, R])  # Coordinates of the center
        self.original_image, self.original_mask = self.create_normalised_target(R)
        self.transformed_image=self.original_image.copy()
        self.transformed_mask=self.original_mask.copy()

    def create_normalised_target(self, R):
        """
        Creates a normalized target with concentric circles of different colors.

        :param R: The radius of the largest circle.
        :return: A tuple (image, mask) where image is the target image and mask is the corresponding mask.
        """
        normalised_target = np.zeros((2*R, 2*R, 3), dtype=np.uint8)
        normalised_mask = np.zeros((2*R, 2*R), dtype=np.uint8)
        centre = self.centre  # Use the center from the class
        radii = np.array([R, 0.8*R, 0.6 * R, 0.4 * R, 0.2 * R])

        # Draw concentric circles on the target image
        for i, radius in enumerate(radii):
            color = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)][i]
            cv2.circle(normalised_target, tuple(centre), int(radius), color, -1)
            cv2.circle(normalised_mask, tuple(centre), int(radius), (1), -1)  # Masking the circles
        return normalised_target, normalised_mask

    def scale_target(self, scale_x, scale_y=None):
        if scale_y is None:
            scale_y = scale_x  # If scale_y is not provided, set it equal to scale_x
        """
        Scales the target by the given scale factors in x and y directions.
        The canvas is resized accordingly, and the center is recalculated.
        It uses the current image and mask size for scaling.

        :param scale_x: Scaling factor in the x direction.
        :param scale_y: Scaling factor in the y direction.
        """
        # Get current dimensions of the transformed target (not the original)
        current_width = self.transformed_image.shape[1]
        current_height = self.transformed_image.shape[0]

        # Calculate the new width and height based on the scaling factors
        new_width = int(current_width * scale_x)
        new_height = int(current_height * scale_y)
        
        # Resize the target image and mask based on the new dimensions
        scaled_target = cv2.resize(self.transformed_image, None , fx=scale_x,fy=scale_y)
        scaled_mask = cv2.resize(self.transformed_mask, None , fx=scale_x,fy=scale_y)
        
        # Recalculate the center of the target after scaling
        self.centre = np.array([self.centre[0]*scale_x, self.centre[1]*scale_y])
        
        # Update the target and mask with the scaled versions
        self.transformed_image = scaled_target
        self.transformed_mask = scaled_mask

# Example usage:
if __name__ == "__main__":
    R = 250  # Example radius size for the target
    target = ArcheryTarget(R)
    # Scale the target by 1.5x in both x and y directions
        # # Display the original target
    # cv2.imshow("Original Archery Target", target.original_image)

    # Define the  homography matrix (3x3)
    H = np.matrix([[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 1]], dtype=np.float32)

    # Get the size of the image
   # height, width, _ = img1.shape

    # Apply the identity homography to the image (it will not change the image)
    original_image = target.original_image
    img_transformed = cv2.warpPerspective(target.original_image, H, target.original_mask.shape)

    # # Display the scaled target
    # cv2.imshow("Scaled Archery Target", target.transformed_image)
    # cv2.imshow("Scaled Target Mask", target.transformed_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
