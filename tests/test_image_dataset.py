import torch
import context # pylint: disable=unused-import
from datasets.image_dataset import ImageDataset

def test_image_dataset():
    params = {
        "dataroot"      : "data/demo",
        "isTrain"       : True,
        "image_size"    : 256,
        "image_depth"   : 3,
    }

    dataset = ImageDataset(**params)
    data = dataset[0]

    image_size = params["image_size"]
    image = data["image"]
    assert torch.is_tensor(image)
    assert image.shape == torch.Tensor(3, image_size, image_size).shape
    assert 0 <= image.min() < 0.5 and 0.5 < image.max() <= 1
    assert isinstance(image, torch.FloatTensor)

    image_path = data["path"]
    assert isinstance(image_path, str)

if __name__ == "__main__":
    test_image_dataset()
