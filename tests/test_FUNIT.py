import torch
import context # pylint: disable=unused-import
from models.FUNIT import FUNIT

def test_FUNIT():
    def fake_batch():
        return torch.randn(2, 3, 128, 128, dtype=torch.float32) * 2 - 1

    net = FUNIT()
    x = fake_batch()
    y1 = fake_batch()
    y2 = fake_batch()

    res = net(x, [y1, y2])

    assert res.shape == fake_batch().shape
    assert -1 <= res.min() < 0 and 0 < res.max() <= 1


if __name__ == "__main__":
    test_FUNIT()
