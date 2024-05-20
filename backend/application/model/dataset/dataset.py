import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self, data_file: str):
        super().__init__()
        with open(data_file, encoding="utf8") as file:
            self.data = json.load(file)

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        # set the tokenizer pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.sys_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a two-sentence horror storyteller.
        Given the user's one sentence, continue with exactly one sentence to make a two-sentence horror story.
        When the user inputs more than one sentence, prompt the user to use one sentence only and do not do anything else.<|eot_id|>"""

        self.user_prompt_format = "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
        self.assistant_format = "<|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"

    def __getitem__(self, index) -> tuple:
        inp = f"{self.sys_prompt}{self.user_prompt_format.format(self.data[index][0])}"
        out = inp + self.assistant_format.format(self.data[index][1])

        out = self.tokenizer(
            out, return_tensors='pt', padding="max_length", max_length=1024, truncation=True
        )

        return {
            'input_ids': out['input_ids'].squeeze(),  # Convert to 1D tensor
            'attention_mask': out['attention_mask'].squeeze(),  # Convert to 1D tensor
            'labels': out['input_ids'].squeeze(),
        }

    def __len__(self):
        return len(self.data)
