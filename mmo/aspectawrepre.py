import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        # Get the dimensions of the input image.
        (h, w) = image.shape[:2]

        # Initialize delta values for cropping.
        dW = 0
        dH = 0

        # Determine whether to resize along width or height.
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.interpolation)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.interpolation)
            dW = int((image.shape[1] - self.width) / 2.0)

        # Get the updated dimensions after resizing.
        (h, w) = image.shape[:2]

        # Crop the image based on the calculated deltas.
        image = image[dH:h - dH, dW:w - dW]

        # Resize the image to the provided spatial dimensions for consistency.
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
