from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D
from PIL import Image

import numpy as np


def to_rgb(X, width=299, height=299):
    """
    This is a generic function to convert greyscale to RGB images,
    but it also resizes to the specified width and height.

    Note that 299x299 with 3 channels is the expected input format for Inception V3.
    X: image arrays
    width
    """
    converted = []
    for x in X:
        image = Image.fromarray(x.astype('uint8'), 'L') #greyscale image
        image = image.convert('RGB')
        image = np.array(image.resize((width,height)))
        converted.append(image)

    return np.array(converted)