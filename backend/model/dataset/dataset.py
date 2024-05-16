import json
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_file: str):
        super().__init__()
        with open(data_file, encoding="utf8") as file:
            self.data = json.load(file)

        self.sys_prompt = """"""
