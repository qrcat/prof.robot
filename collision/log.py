from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class RegistorLog:
    def __init__(self, *args):
        self.logers = args

    def log(self, ordered_dict: dict, step: int, phase: str):
        for loger in self.logers:
            if isinstance(loger, tqdm):
                if phase == "train":
                    loger.set_postfix(ordered_dict)
                else:
                    loger.set_description(
                        " ".join(
                            [f"{key}:{item}" for key, item in ordered_dict.items()]
                        )
                    )
            elif isinstance(loger, SummaryWriter):
                for key, value in ordered_dict.items():
                    loger.add_scalar(f"{phase}/{key}", value, step)

    def log_test(self, ordered_dict: dict, step: int):
        self.log(ordered_dict, step, "test")

    def log_train(self, ordered_dict: dict, step: int):
        self.log(ordered_dict, step, "train")
