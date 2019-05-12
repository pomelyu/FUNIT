from pathlib import Path
import urllib.request
import warnings
import hashlib
from skimage import io, color
import numpy as np

blank_image = np.zeros((224, 224), dtype=np.uint8)

class ResouceInvalidError(Exception):
    """Raised when the resource is invalid"""
    def __init__(self, message):
        Exception.__init__(self, message)


def valid_image(path, min_size=10):
    min_size = (min_size << 10) # convert min_size from KB to B
    if Path(path).stat().st_size < min_size:
        return False
    return True

def read_image_url(path, gray_scale=False):
    try:
        image = io.imread(path)
    except urllib.error.HTTPError as err:
        raise ResouceInvalidError(err.code)
    except Exception as e:
        raise ResouceInvalidError(e)

    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if len(image.shape) > 3:
        raise ResouceInvalidError("Invalid shape {}".format(image.shape))

    if gray_scale:
        image = color.rgb2gray(image)
    else:
        image = color.gray2rgb(image)

    return image

def read_image(path, min_size=10, gray_scale=False):
    if not Path(path).exists():
        raise ResouceInvalidError("image not found")

    if not valid_image(path, min_size=min_size):
        raise ResouceInvalidError("image too small")

    image = io.imread(path)

    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if gray_scale:
        image = color.rgb2gray(image)
    else:
        image = color.gray2rgb(image)

    return image


def save_image(path, image):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(path, image)

def save_blank(path):
    save_image(path, blank_image)

def hash_string(the_str):
    return hashlib.sha256(the_str.encode("utf-8")).hexdigest()
