"""Base DataModule class."""
import cv2

import FSDL.plate_recognizer.utils as utils

from enum import Enum

IMAGE_SIZE = 224
DATA_ROOT = "/content/drive/MyDrive/data/DataCentric"


class DataType(Enum):
    Train = 1
    Val = 2
    Test = 3


class BaseData():
    def __init__(self, X_path=DATA_ROOT + "/train/", Y_path=DATA_ROOT + "/train") -> None:
        self.X_path = X_path
        self.Y_path = Y_path
        self.X_raw = None
        self.Y_raw = None
        self.X = None
        self.Y = None

    def load_images(self, path):
        paths = utils.glob_files(path)

        X_raw = []
        for file in paths:
            image = cv2.imread(file)
            image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
            X_raw.append(np.array(image))

        return np.array(X_raw)