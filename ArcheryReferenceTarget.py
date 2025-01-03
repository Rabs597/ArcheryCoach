import numpy as np
import cv2

# scaling_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])  # Use single scale for both x and y


class ArcheryReferenceTarget:
    def __init__(self, R=100):
        self.original_radius = R  # Store the original radius
        self.centre = np.array([R, R])  # Coordinates of the center
        self.original_image, self.original_mask = self.create_normalised_target(R)
        self.transformed_image = self.original_image.copy()
        self.transformed_mask = self.original_mask.copy()

    def create_normalised_target(self, R):
        """
        Creates a normalized target with concentric circles of different colors.

        :param R: The radius of the largest circle.
        :return: A tuple (image, mask) where image is the target image and mask is the corresponding mask.
        """
        normalised_target = np.zeros((2 * R, 2 * R, 3), dtype=np.uint8)
        normalised_mask = np.zeros((2 * R, 2 * R), dtype=np.uint8)
        centre = self.centre  # Use the center from the class
        radii = np.array([R, 0.8 * R, 0.6 * R, 0.4 * R, 0.2 * R])

        # Draw concentric circles on the target image
        for i, radius in enumerate(radii):
            color = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)][i]
            cv2.circle(normalised_target, tuple(centre), int(radius), color, -1)
            cv2.circle(normalised_mask, tuple(centre), int(radius), (1), -1)  # Masking the circles
        return normalised_target, normalised_mask

    def apply_homography(self, H):
        """
        Applies the homography matrix to the image after translating the center to the origin.
        After the transformation, it translates the points back to the original center.
        
        :param H: Homography matrix.
        """
        # Step 1: Translate the center of the image to the origin
        translation_matrix_to_origin = np.array([[1, 0, -self.centre[0]], 
                                                 [0, 1, -self.centre[1]], 
                                                 [0, 0, 1]], dtype=np.float32)
        
        

        # Step 2: Apply the homography transformation
        H_about_origin = H @ translation_matrix_to_origin #@ H #@ np.linalg.inv(translation_matrix_to_origin)
       # H_about_origin = translation_matrix_to_origin @ H #@ H #@ np.linalg.inv(translation_matrix_to_origin)

        # Step 3: Calculate the current corner points of the image
        height, width = self.transformed_image.shape[:2]
       
        #points are the original centre and the four corners
        points = np.array([[self.centre[0],self.centre[1]],[0, 0], [width , 0], [0, height ], [width , height ]], dtype=np.float32)
        transformed_points = cv2.perspectiveTransform(np.array([points]), H_about_origin)[0]

        # Get the new bounding box for the transformed image
        x_min = int(np.min(transformed_points[:, 0]))
        x_max = int(np.max(transformed_points[:, 0]))
        y_min = int(np.min(transformed_points[:, 1]))
        y_max = int(np.max(transformed_points[:, 1]))
        newwidthoftarget=x_max-x_min
        newheightoftarget=y_max-y_min

        # Store the new position of the original center after the homography transformation
        self.centre = transformed_points[0]+[(newwidthoftarget)/2, (newheightoftarget)/2]

        # Calculate the translation required to move the new center to the original center
        translation_matrix_back = np.array([[1, 0, self.centre[0]],
                                            [0, 1, self.centre[1]],
                                            [0, 0, 1]], dtype=np.float32)

        #resize the canvas for both the image and the mask
        newshape=np.array([newwidthoftarget, newheightoftarget])+transformed_points[0]*2
        newshape=newshape.astype(int)
        self.transformed_image=np.zeros([newshape[0],newshape[1],3])
        self.transformed_mask=np.zeros([newshape[0],newshape[1]])
        
        # Apply the homography and translation back
        H_final = translation_matrix_back @ H_about_origin

        # Apply the final homography matrix to the image
        self.transformed_image = cv2.warpPerspective(self.original_image, H_final, newshape)

        # Adjust the transformed mask similarly (if required)
        self.transformed_mask = cv2.warpPerspective(self.original_mask, H_final, newshape)


# Example usage:
if __name__ == "__main__":
    R = 100  # Example radius size for the target
    target = ArcheryReferenceTarget(R)

    # Define a scaling homography matrix (for example, scaling by 2x)
    H = np.matrix([[2, 0, 20],
                  [0.2, 2, 0],
                  [0, 0, 1]], dtype=np.float32)

    # Apply the homography with the method
    target.apply_homography(H)
    testimage=target.transformed_image
    # Display the transformed target
    cv2.imshow("Transformed Archery Target", target.transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x=1


## Notes
# at present the Homography matrix is scaling the translation, which I didn't expect.  
# This might be correct but it has the effect of moving the image off the canvas.
# Either need to change the canvas resize by double the amount, or fix the martix operation