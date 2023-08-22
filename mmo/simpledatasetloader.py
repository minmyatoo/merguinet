import numpy as np
import cv2
import os
import logging

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        """
        Initialize the SimpleDatasetLoader.

        Args:
            preprocessors (list, optional): A list of image preprocessors. Defaults to None.
        """
        self.preprocessors = preprocessors if preprocessors else []

    def load(self, imagePaths, verbose=-1):
        """
        Load images from file paths and apply preprocessors.

        Args:
            imagePaths (list): A list of file paths to images.
            verbose (int, optional): Verbosity level. Defaults to -1.

        Returns:
            tuple: A tuple containing NumPy arrays for data (images) and labels.
        """
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            try:
                # Load the image and extract the class label assuming the path format:
                # /path/to/dataset/{class}/{image}.jpg
                image = cv2.imread(imagePath)
                label = os.path.basename(os.path.dirname(imagePath))

                # Apply preprocessors
                for p in self.preprocessors:
                    image = p.preprocess(image)

                # Append the processed image to data and label to labels
                data.append(image)
                labels.append(label)

                # Show an update every `verbose` images
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    logging.info("[INFO] Processed {}/{}".format(i + 1, len(imagePaths)))

            except Exception as e:
                logging.error(f"[ERROR] Failed to load/process image: {imagePath}. Error: {e}")
                continue  # Continue processing other images even if one fails

        return (np.array(data), np.array(labels))
