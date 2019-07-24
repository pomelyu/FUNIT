from pathlib import Path
import argparse
import arrow
import torch
from tqdm import tqdm
import numpy as np
import gin

import gin_pytorch  # pylint: disable=unused-import
import models       # pylint: disable=unused-import
from datasets import create_dataloader
from logger import Logger


@gin.configurable
def train(
        checkpoints_dir=None, exp_name=None, model=None, train_dataset=None, valid_dataset=None, \
        use_gpu=True, epoch_stable=10, epoch_trans=10, epoch_start=1, load_epoch=None, \
        print_steps=500, display_steps=2000, \
        save_model_epoch=1, save_latest_steps=50000 \
    ):
    assert checkpoints_dir and exp_name and model and train_dataset

    experiment_dir = Path(checkpoints_dir) / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Prepare for Training
    logger = Logger(experiment_dir, only_train=(valid_dataset is None))
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model(isTrain=True, device=device)
    train_dataloader = create_dataloader(train_dataset(isTrain=True), shuffle=True)
    valid_dataloader = create_dataloader(valid_dataset(isTrain=False), shuffle=False) if valid_dataset else None

    # Copy the config file and print params
    timestamp = arrow.utcnow().to("local").format("[YYYY_MMDD] HH'mm'ss")
    exp_config = experiment_dir / "config_{}.gin".format(timestamp)
    exp_config.write_text(Path(opt.config).read_text())
    exp_params = experiment_dir / "params_{}.txt".format(timestamp)
    exp_params.write_text(gin.operative_config_str())

    if load_epoch:
        model.load_networks(experiment_dir, load_epoch)

    epoch_total = epoch_stable * 4 + epoch_trans * 3
    pg_indexes = [0] * epoch_stable + [(i+1)/epoch_trans for i in range(epoch_trans)] + \
                    [1] * epoch_stable + [1 + (i+1)/epoch_trans for i in range(epoch_trans)] + \
                    [2] * epoch_stable + [2 + (i+1)/epoch_trans for i in range(epoch_trans)] + \
                    [3] * epoch_stable

    steps = (epoch_start - 1) * len(train_dataloader)
    for epoch in range(epoch_start, epoch_total+1):
        # Training
        train_losses = {k: [] for k in model.losses}

        pg_index = pg_indexes[epoch - 1]
        model.set_pg_index(pg_index)
        print("progressive index: {:.5f}".format(pg_index))
        for data in tqdm(train_dataloader, total=len(train_dataloader), ascii=True):
            model.set_input(data)
            model.forward()
            model.optimize_parameters()

            for k, v in model.get_losses().items():
                train_losses[k].append(v)

            steps += 1
            if steps % print_steps == 0:
                train_message = ""
                for k, v in model.get_losses().items():
                    logger.add_scalar(k, v, steps, phase="train")
                    train_message += "{}: {:.4f}, ".format(k, v)
                train_message = "[Epoch {:0>3d}, {:0>6d}] train - {}".format(epoch, steps, train_message)
                tqdm.write(train_message)

            if steps % display_steps == 0:
                for tag, image in model.get_visuals().items():
                    tag_arr = tag.split("_")
                    logger.add_images("/".join(tag_arr), image, steps)

            if steps % save_latest_steps == 0:
                model.save_networks(experiment_dir, "latest")

        model.update_epoch(epoch)
        if epoch % save_model_epoch == 0:
            model.save_networks(experiment_dir, epoch)
        model.save_networks(experiment_dir, "latest", with_optimizer=True)

        # Log training loss
        train_message = ""
        for k, v in train_losses.items():
            train_message += "{}: {:.4f}, ".format(k, np.mean(v))
        train_message = "[Epoch {:0>3d}] train - {}".format(epoch, train_message)

        # Validation
        if valid_dataloader is None:
            logger.save_message(train_message)
            continue

        valid_losses = {k: [] for k in model.losses}
        with torch.no_grad():
            for data in tqdm(valid_dataloader, total=len(valid_dataloader), ascii=True):
                model.set_input(data)
                model.forward()
                model.evaluate()

                for k, v in model.get_losses().items():
                    valid_losses[k].append(v)

        # Log validation loss
        valid_message = ""
        for k, v in valid_losses.items():
            logger.add_scalar(k, np.mean(v), steps, phase="valid")
            valid_message += "{}: {:.4f}, ".format(k, np.mean(v))
        valid_message = "[Epoch {:0>3d}] valid - {}".format(epoch, valid_message)
        logger.save_message(train_message)
        logger.save_message(valid_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=str)
    opt, _ = parser.parse_known_args()

    gin.parse_config_file(opt.config)

    # Start training
    train()
