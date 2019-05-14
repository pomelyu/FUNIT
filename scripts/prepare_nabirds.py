import argparse
from pathlib import Path
import pandas as pd
from skimage import transform
from skimage import io
import numpy as np
from tqdm import tqdm

def preprocess_nabirds(image_file, nabirds_dir, result_dir, output_size):
    nabirds_dir = Path(nabirds_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)

    df_image = read_csv(image_file, ["path"])
    df_bbox = read_csv(nabirds_dir / "bounding_boxes.txt", ["x", "y", "w", "h"])
    df_class = read_csv(nabirds_dir / "image_class_labels.txt", ["class_id"])
    df_combined = df_image.join(df_bbox).join(df_class)

    for _, info in tqdm(df_combined.iterrows(), total=df_combined.shape[0], ascii=True):
        result_folder = result_dir / "{:0>4d}".format(info["class_id"])
        result_folder.mkdir(exist_ok=True, parents=True)

        image = io.imread(nabirds_dir / "images" / info["path"])
        bbox = np.array([info["x"], info["y"], info["w"], info["h"]], dtype=int)
        thumbnail = crop_bbox(image, bbox, output_size)
        io.imsave(result_dir / info["path"], thumbnail)

def read_csv(path, columns):
    df = pd.read_csv(path, sep=" ", header=None, index_col=0)
    df.columns = columns
    return df

def crop_bbox(image, bbox, output_size):
    ih, iw = image.shape[:2]

    x, y, w, h = bbox
    d = max(w, h)

    res = np.zeros((d, d, 3), dtype=np.uint8)
    dx1 = max(0, x + (w - d) // 2)
    dy1 = max(0, y + (h - d) // 2)
    dx2 = min(iw, x + (w + d) // 2)
    dy2 = min(ih, y + (h + d) // 2)

    rx1 = -min(0, x + (w - d) // 2)
    ry1 = -min(0, y + (h - d) // 2)
    rx2 = d - max(0, x + (w + d) // 2 - iw)
    ry2 = d - max(0, y + (h + d) // 2 - ih)

    res[ry1:ry2, rx1:rx2] = image[dy1:dy2, dx1:dx2]
    res = transform.resize(res, (output_size, output_size))
    res = (res * 255).astype(np.uint8)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_file", type=str)
    parser.add_argument("--nabirds_dir", type=str, default="data/nabirds")
    parser.add_argument("--result_dir", type=str, default="data/nabirds/thumbnails")
    parser.add_argument("--output_size", type=int, default=128)
    opt, _ = parser.parse_known_args()

    preprocess_nabirds(opt.image_file, opt.nabirds_dir, opt.result_dir, opt.output_size)
