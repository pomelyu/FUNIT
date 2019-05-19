from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from imgaug import augmenters as iaa
import gin
from datasets.augmenters import PadToSquare, Normalize, ValueClip, BGRtoRGB, ToTensor
from utils import read_image

IMAGE_EXT = [".jpg", ".png"]

@gin.configurable('Image2ImageDataset', blacklist=["isTrain"])
def createDataset(dataroot, isTrain, class_file, image_size=128, image_depth=3, num_class=444):
    return Image2ImageDataset(dataroot, isTrain, class_file, image_size, image_depth, num_class)

class Image2ImageDataset(Dataset):
    def __init__(self, dataroot, isTrain, class_file, \
                    image_size=128, image_depth=3, num_class=444):

        assert image_depth in [1, 3]

        self.isTrain = isTrain
        self.gray_scale = (image_depth == 1)

        dataroot = Path(dataroot)
        classe_ids = np.loadtxt(class_file, dtype=int)
        self.data = []
        self.class_indexes = []
        self.class_offset = [0]
        for i, class_id in enumerate(classe_ids[:num_class]):
            image_paths = images_in_dir(dataroot / "{:0>4d}".format(class_id))
            self.data += image_paths
            self.class_indexes += [i] * len(image_paths)
            self.class_offset.append(self.class_offset[-1] + len(image_paths))

        if not self.data:
            raise IOError("Found 0 images in {}, image must has .png or .jpg suffix".format(dataroot))

        self.transform = iaa.Sequential([
            PadToSquare(),
            iaa.Resize({"height": image_size, "width": image_size}),
            BGRtoRGB(),
            Normalize(mean=127.5, var=127.5),
            ValueClip(-1, 1),
            ToTensor(),
        ])

    def __getitem__(self, index):
        class_id = self.class_indexes[index]
        random_offset = np.random.randint(self.class_offset[class_id], self.__len__() + self.class_offset[class_id - 1])
        target_idx = random_offset % self.__len__()
        target_class = self.class_indexes[target_idx]

        source = read_image(self.data[index], min_size=0, gray_scale=self.gray_scale)
        target = read_image(self.data[target_idx], min_size=0, gray_scale=self.gray_scale)

        return {
            "source": self.transform.augment_image(source),
            "target": self.transform.augment_image(target),
            "source_label": torch.ones(1, dtype=torch.int) * class_id,
            "target_label": torch.ones(1, dtype=torch.int) * target_class,
        }


    def __len__(self):
        return self.class_offset[-1]

def images_in_dir(dir_path):
    return [p for p in dir_path.iterdir() if p.suffix in IMAGE_EXT]
