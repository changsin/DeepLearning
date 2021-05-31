"""Kaggle DataModule"""
from lxml import etree
from matplotlib import pyplot as plt

import argparse
import cv2
import glob
import json
import numpy as np
import os

# NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
# from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order

# opener = urllib.request.build_opener()
# opener.addheaders = [("User-agent", "Mozilla/5.0")]
# urllib.request.install_opener(opener)

IMAGE_SIZE = 224
DATA_ROOT = "/content/drive/MyDrive/data/Kaggle_license_plates"

# class KaggleData(BaseDataModule):
class KaggleData():
    """
    """
    def __init__(self, X_path=DATA_ROOT + "/images/", Y_path=DATA_ROOT + "/annotations.json") -> None:
        self.X_path = X_path
        self.Y_path = Y_path
        self.X_raw = None
        self.Y_raw = None
        self.X = None
        self.Y = None

    def prepare_data(self, *args, **kwargs) -> None:
        self.X_raw = self.load_images(DATA_ROOT + "/images/")
        self.Y_raw = self.load_labels(DATA_ROOT + "/annotations.json")
        self.X, self.Y = self.normalize(self.X_raw, self.Y_raw)

    # def setup(self, stage=None) -> None:

    def to_json(self, path, data):
        """
        save json data to path
        """
        # y_yolov5_lists = y_yolov5.tolist()
        # json_str = json.dumps(y_yolov5_lists)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def from_json(self, path):
        """
        save json data to path
        """
        file = open(path, 'r', encoding='utf-8')
        return json.load(file)

    def resize_annotation(self, f):
        tree = etree.parse(f)
        for dim in tree.xpath("size"):
            width = int(dim.xpath("width")[0].text)
            height = int(dim.xpath("height")[0].text)
        for dim in tree.xpath("object/bndbox"):
            xmin = int(dim.xpath("xmin")[0].text)/(width/IMAGE_SIZE)
            ymin = int(dim.xpath("ymin")[0].text)/(height/IMAGE_SIZE)
            xmax = int(dim.xpath("xmax")[0].text)/(width/IMAGE_SIZE)
            ymax = int(dim.xpath("ymax")[0].text)/(height/IMAGE_SIZE)

        # y_yolov5 = np.array([to_yolov5(y) for y in y_train_raw])
        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    def extract_annotations(self, label_file, class_id):
        labels = []
        with open(label_file, "r") as file:
            count = 0
            for line in file:
                tokens = [float(token) for token in line.split()]
                if tokens[0] == class_id:
                    count += 1
                    # print(line)
                    labels.append(np.array(tokens[1:]))

            if count > 1:
                print("WARNING: More than one license plate was found: ", count, label_file)
            elif count == 0:
                print("WARNING: No license plate was found: ", count, label_file)

        return np.array(labels)

    def to_yolov5(self, y):
        """
        # change to yolo v5 format
        # https://github.com/ultralytics/yolov5/issues/12
        # [x_top_left, y_top_left, x_bottom_right, y_bottom_right] to
        # [x_center, y_center, width, height]
        """
        width = y[2] - y[0]
        height = y[3] - y[1]

        if width < 0 or height < 0:
            print("ERROR: negative width or height ", width, height, y)
            raise AssertionError("Negative width or height")
        return int(y[0] + (width/2)), int(y[1] + (height/2)), width, height

    def load_images(self, path):
        path = os.path.join(path,'*')
        files = glob.glob(path)
        # We sort the images in alphabetical order to match them
        #  to the annotation files
        files.sort()

        X_raw = []
        for file in files:
            image = cv2.imread(file)
            image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
            X_raw.append(np.array(image))

        return np.array(X_raw)

    def load_labels(self, path, yolov5=True):
        """
        if path is an annotation file, just load it
        and return it immediately to save time
        """
        if path.endswith('.json'):
            return self.from_json(path)

        path = os.path.join(path,'*')
        files = glob.glob(path)
        files.sort()

        y_raw = []
        for file in files:
            if yolov5:
                y_raw.append(self.extract_annotations(file, 0))
            else:
                y_raw.append(self.resize_annotation(file))
        return np.array(y_raw)

    # transform to arrays and normalize
    def normalize(self, X_raw, y_raw):
        X = np.array(X_raw)
        y = np.array(y_raw)
        y = y.reshape((y.shape[0], -1))

        #  Renormalisation
        X = X / IMAGE_SIZE
        y = y / IMAGE_SIZE

        return X, y

# if __name__ == "__main__":
    # load_and_print_info(MNIST)