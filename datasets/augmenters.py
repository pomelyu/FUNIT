import numpy as np
import imgaug as ia
from imgaug.augmenters.meta import Augmenter
import torch

class PadToSquare(Augmenter):
    def __init__(self, padding_value=0, name=None, deterministic=False, random_state=None):
        super(PadToSquare, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.padding_value = padding_value

    def _augment_images(self, images, random_state, parents, hooks):
        results = []
        for image in images:
            h, w = image.shape[:2]
            if h == w:
                results.append(image)
                continue

            dim = max(h, w)
            oh = (dim - h) // 2
            ow = (dim - w) // 2
            if len(image.shape) == 2:
                padded = np.ones((dim, dim)) * self.padding_value
            else:
                padded = np.ones((dim, dim, image.shape[2])) * self.padding_value
            padded[oh:oh+h, ow:ow+w] = image
            results.append(padded)

        return results

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError()

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        results = []
        for kps in keypoints_on_images:
            shape = list(kps.shape)
            h, w = shape[:2]
            if h == w:
                results.append(kps)
                continue

            dim = max(h, w)
            oh = (dim - h) // 2
            ow = (dim - w) // 2
            shape[:2] = [dim, dim]
            padded_kps = ia.KeypointsOnImage([
                ia.Keypoint(x=(kp.x+ow), y=(kp.y+oh)) for kp in kps.keypoints
            ], shape=shape)

            results.append(padded_kps)

        return results

    def get_parameters(self):
        return [self.padding_value]


class ToTensor(Augmenter):
    def _augment_images(self, images, random_state, parents, hooks):
        results = [torch.from_numpy(np.moveaxis(image.astype(np.float32), -1, 0)) for image in images]
        return results

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise TypeError("Can't apply ToTensor to heatmaps")

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise TypeError("Can't apply ToTensor to keypoints")

    def get_parameters(self):
        return []


class Normalize(Augmenter):
    def __init__(self, mean, var, name=None, deterministic=False, random_state=None):
        super(Normalize, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.mean = np.array(mean).reshape((1, 1, -1))
        self.var = np.array(var).reshape((1, 1, -1))

    def _augment_images(self, images, random_state, parents, hooks):
        results = [(image - self.mean) / self.var for image in images]
        return results

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError()

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise TypeError("Can't apply Normalize to keypoints")

    def get_parameters(self):
        return [self.mean, self.var]


class ValueClip(Augmenter):
    def __init__(self, min_value=0, max_value=1, name=None, deterministic=False, random_state=None):
        super(ValueClip, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.min_value = min_value
        self.max_value = max_value

    def _augment_images(self, images, random_state, parents, hooks):
        results = [np.clip(image, self.min_value, self.max_value) for image in images]
        return results

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        results = [np.clip(image, self.min_value, self.max_value) for image in images]
        return results

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise TypeError("Can't apply ValueClip to keypoints")

    def get_parameters(self):
        return [self.min_value, self.max_value]


class BGRtoRGB(Augmenter):
    def _augment_images(self, images, random_state, parents, hooks):
        results = []
        for image in images:
            if len(image.shape) == 3:
                image = image[..., ::-1]
            results.append(image)
        return results

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise TypeError("Can't apply BGRtoRGB to heatmaps")

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise TypeError("Can't apply BGRtoRGB to keypoints")

    def get_parameters(self):
        return []
