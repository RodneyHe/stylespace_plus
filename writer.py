from general_utils.general_utils import convert_tensor_to_image
from pathlib import Path
import torch

from torch.utils.tensorboard import SummaryWriter

class Writer(object):
    writer = None

    @staticmethod
    def set_writer(results_dir):
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        Writer.writer = SummaryWriter(str(results_dir))

    @staticmethod
    def add_scalar(tag, val, step):
        if isinstance(val, torch.Tensor):
            val = val.item()

        Writer.writer.add_scalar(tag, val, step)

    # @staticmethod
    # def add_image(tag, val, step):
    #     val = convert_tensor_to_image(val)

    #     if tb.rank(val) == 3:
    #         val = tb.expand_dims(val, 0)

    #     with Writer.writer.as_default():
    #         tb.summary.image(tag, val, step)

    # @staticmethod
    # def flush():
    #     with Writer.writer.as_default():
    #         Writer.writer.flush()