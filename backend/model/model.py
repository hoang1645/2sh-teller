from typing import Callable, Literal, Tuple, Union

import torch
from bitsandbytes.optim import PagedAdamW8bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Llama3Model:
    def __init__(
        self,
        model: Literal['8B', '70B', '8B-Instruct', '70B-Instruct'] = "8B-Instruct",
        load_in_n_bits: Literal[4, 8, 16, 32] = 4,
        efficient_finetuning_method: Literal["lora", "qlora", "reft", None] = None,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.98),
        weight_decay: float = 1e-3,
        eps: float = 1e-8,
    ):
        """
        Wrapper for Llama 3 model, including training and inference.
        """

        self.trainable = True
        if load_in_n_bits <= 8 and efficient_finetuning_method is None:
            self.trainable = False
        if load_in_n_bits <= 8 and efficient_finetuning_method == "reft":
            raise ValueError(
                f"found load_in_n_bits = {load_in_n_bits} "
                "and efficient_finetuning_method = {efficient_finetuning_method}"
            )

        # quantization configurations
        quantization_config = {"load_in_4bit": False, "load_in_8bit": False}
        if load_in_n_bits == 4:
            quantization_config["load_in_4bit"] = True
            quantization_config["quantization_config"] = BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                **quantization_config,
            )
            quantization_config["torch_dtype"] = torch.bfloat16
        if load_in_n_bits == 8:
            quantization_config["load_in_8bit"] = True
            quantization_config["quantization_config"] = BitsAndBytesConfig(
                llm_int8_skip_modules="lm_head", **quantization_config
            )
            quantization_config["torch_dtype"] = torch.bfloat16
        else:
            quantization_config['torch_dtype'] = torch.float32

        self.model_id = f"meta-llama/Meta-Llama-3-{model}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM(self.model_id, **quantization_config)

        self.optimizer = self.__configure_optimizer(
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, **quantization_config
        )

    def __optimizer(
        self, load_in_4bit: bool, load_in_8bit: bool
    ) -> Callable[..., torch.optim.Optimizer]:
        if load_in_4bit or load_in_8bit:
            return PagedAdamW8bit
        return torch.optim.AdamW

    def __configure_optimizer(
        self,
        load_in_4bit: bool,
        load_in_8bit: bool,
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float,
        eps: float,
        **unused_kwargs,
    ):
        return self.__optimizer(load_in_4bit, load_in_8bit)(
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps
        )
