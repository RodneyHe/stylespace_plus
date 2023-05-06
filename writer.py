from general_utils.general_utils import convert_tensor_to_image, concatenate_image
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

    @staticmethod
    def add_image(tag, val, step):
        if isinstance(val, list):
            cat_image = concatenate_image(val)
            val = convert_tensor_to_image(cat_image)
        else:
            val = convert_tensor_to_image(val)

        Writer.writer.add_image(tag, val, step)

    @staticmethod
    def add_graph(model, *graph_inputs):
        Writer.writer.add_graph(model, *graph_inputs)

    @staticmethod
    def flush():
        Writer.writer.flush()