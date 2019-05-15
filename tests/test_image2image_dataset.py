import torch
import context # pylint: disable=unused-import
from datasets.image2image_dataset import Image2ImageDataset

def test_image2image_dataset():
    params = {
        "dataroot"      : "data/nabirds_mini/thumbnails/",
        "class_file"    : "data/nabirds_mini/train.txt",
        "isTrain"       : True,
        "image_size"    : 128,
        "image_depth"   : 3,
        "num_class"     : 80,
    }

    dataset = Image2ImageDataset(**params)
    data = dataset[0]

    image_size = params["image_size"]
    source = data["source"]
    assert torch.is_tensor(source)
    assert source.shape == torch.Tensor(3, image_size, image_size).shape
    assert 0 <= source.min() < 0.5 and 0.5 < source.max() <= 1
    assert isinstance(source, torch.FloatTensor)

    target = data["target"]
    assert torch.is_tensor(target)
    assert target.shape == torch.Tensor(3, image_size, image_size).shape
    assert 0 <= target.min() < 0.5 and 0.5 < target.max() <= 1
    assert isinstance(target, torch.FloatTensor)

    source_label = data["source_label"]
    target_label = data["target_label"]
    assert source_label != target_label

if __name__ == "__main__":
    test_image2image_dataset()
