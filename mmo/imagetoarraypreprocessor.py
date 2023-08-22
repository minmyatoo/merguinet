from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    """
    A preprocessor to convert images to NumPy arrays.

    Args:
        dataFormat (str, optional): The image data format ('channels_last' or 'channels_first').
            Defaults to None.

    Attributes:
        dataFormat (str): The image data format to be used for array conversion.
    """

    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        """
        Preprocesses an image by converting it to a NumPy array.

        Args:
            image (PIL.Image.Image or numpy.ndarray): The input image to be converted.

        Returns:
            numpy.ndarray: The NumPy array representation of the input image.
        """
        return img_to_array(image, data_format=self.dataFormat)
