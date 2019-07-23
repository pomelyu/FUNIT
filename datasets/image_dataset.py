from pathlib import Path
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
import gin
from datasets.augmenters import PadToSquare, Normalize, ValueClip, BGRtoRGB, ToTensor
from utils import read_image

@gin.configurable('ImageDataset', blacklist=["isTrain"])
def createDataset(dataroot, isTrain, image_size=256, image_depth=3):
    return ImageDataset(dataroot, isTrain, image_size, image_depth)

class ImageDataset(Dataset):
    def __init__(self, dataroot, isTrain, image_size=256, image_depth=3):
        assert dataroot
        assert image_depth in [1, 3]

        self.isTrain = isTrain
        self.gray_scale = (image_depth == 1)
        self.image_size = image_size
        self.data = [f for f in Path(dataroot).iterdir() if f.suffix == ".png" or f.suffix == ".jpg"]

        if not self.data:
            raise IOError("Found 0 images in {}, image must has .png or .jpg suffix".format(dataroot))

        self.transform = iaa.Sequential([
            PadToSquare(),
            iaa.Resize({"height": self.image_size, "width": self.image_size}),
            Normalize(mean=0, var=255),
            ValueClip(0, 1),
            ToTensor(),
        ])

    def __getitem__(self, index):
        f_image = self.data[index]
        image = read_image(f_image, gray_scale=self.gray_scale, min_size=0)
        image = self.transform.augment_image(image)

        return {"image": image, "path": str(f_image.name)}

    def __len__(self):
        return len(self.data)
