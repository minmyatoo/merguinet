import cv2

class SimplePreprocessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        """
        Initialize the SimplePreprocessor.

        Args:
            width (int): The target width for resizing.
            height (int): The target height for resizing.
            interpolation (int, optional): The interpolation method. Defaults to cv2.INTER_AREA.
        """
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        """
        Resize an image to a fixed size.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The resized image.
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
