from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, LoraModel
from bitsandbytes.optim import PagedAdamW8bit
from accelerate import Accelerator
from typing import Literal

class Llama3Model(object):
    def __init__(self, model:Literal['8B', '70B', '8B-Instruct', '70B-Instruct']="8B-Instruct",
                 load_in_n_bytes:Literal[4, 8, 16, 32]=4, efficient_finetuning_method:Literal["lora", "qlora", "reft", ]):
        pass