from pathlib import Path
import time
import torch
from tensorboardX import SummaryWriter

class Logger():
    """Tensorboard logger."""

    def __init__(self, log_dir, only_train=True):
        """Initialize summary writer."""
        self.only_train = only_train

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if only_train:
            train_dir = log_dir
        else:
            train_dir = log_dir / "train"
            valid_dir = log_dir / "valid"
            train_dir.mkdir(parents=True, exist_ok=True)
            valid_dir.mkdir(parents=True, exist_ok=True)
            self.valid_writer = SummaryWriter(log_dir=str(valid_dir))

        self.train_writer = SummaryWriter(log_dir=str(train_dir))
        self.log_path = log_dir / "log.txt"

        now = time.strftime("%c")
        self.save_message("================ Taining Start ({}) ================\n".format(now))

    def add_scalar(self, tag, value, step=None, phase="train"):
        self._get_writer(phase).add_scalar(tag, value, step)

    def add_images(self, tag, images, step, phase="train"):
        self._get_writer(phase).add_images(tag, images, step)

    def add_text(self, tag, text, phase="train"):
        self._get_writer(phase).add_text(tag, text)

    def save_message(self, message):
        print(message)
        with self.log_path.open("a") as log_file:
            log_file.write("{}\n".format(message))

    def _get_writer(self, phase="train"):
        if self.only_train:
            return self.train_writer
        if phase == "train":
            return self.train_writer
        elif phase == "valid":
            return self.valid_writer
        else:
            raise Exception("Unknown phase: {}".format(phase))
