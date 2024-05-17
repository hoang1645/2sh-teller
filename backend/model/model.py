from typing import Callable, Literal, Tuple, List, Any

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from pyreft import get_reft_model, ReftConfig, LoreftIntervention, ReftTrainerForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer as HFTrainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)


# TODO: add comprehensive docstrings for important methods
class Llama3Model:
    def __init__(
        self,
        model: Literal['8B', '70B', '8B-Instruct', '70B-Instruct'] = "8B-Instruct",
        custom_checkpoint_path: str = None,
        load_in_n_bits: Literal[4, 8, 16, 32] = 4,
        efficient_finetuning_method: Literal["lora", "reft", None] = None,
        lora_apply_layers: List[str] = ["self_attn", "mlp", "embed_tokens"],
        lora_dropout: float = 0.1,
        reft_component: int | str = None,
        reft_low_rank_dim: int = None,
    ):
        """
        Wrapper for Llama 3 model, including training and inference.
        """

        # save important args to use in other methods
        self.trainable = True
        self.load_in_nbits = load_in_n_bits
        self.efficient_finetuning_method = efficient_finetuning_method

        # reft currently not working with llama3
        if efficient_finetuning_method == "reft":
            raise ValueError("ReFT and Llama3 do not mix well, yet.")

        # check args
        if load_in_n_bits <= 8 and efficient_finetuning_method is None:
            self.trainable = False
        if load_in_n_bits <= 8 and efficient_finetuning_method == "reft":
            raise ValueError(
                f"found load_in_n_bits = {load_in_n_bits} "
                "and efficient_finetuning_method = {efficient_finetuning_method}"
            )

        # quantization configurations
        quantization_config = {}
        # 4bit (for qlora)
        if load_in_n_bits == 4:
            quantization_config["quantization_config"] = BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                load_in_4bit=True,
            )
            quantization_config["torch_dtype"] = torch.bfloat16
        # 8bit (llm-int8)
        if load_in_n_bits == 8:
            quantization_config["quantization_config"] = BitsAndBytesConfig(
                llm_int8_skip_modules="lm_head", llm_int8_has_fp16_weight=True, load_in_8bit=True
            )
            quantization_config["torch_dtype"] = torch.bfloat16
        # 16bit (bf16)
        if load_in_n_bits == 16:
            quantization_config['torch_dtype'] = torch.bfloat16
        # 32bit (fp32)
        if load_in_n_bits == 32:
            quantization_config['torch_dtype'] = torch.float32
        # model initialization
        self.model_id = f"meta-llama/Meta-Llama-3-{model}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # set the tokenizer end-of-sentence token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            custom_checkpoint_path
            or self.model_id,  # load checkpoint if exists, if not load the baseline
            **quantization_config,
        )
        # optimizer
        self.optimizer = self.__optimizer(**dict(quantization_config['quantization_config']))
        # prepare model for quantization training
        if load_in_n_bits <= 8:
            self.model = prepare_model_for_kbit_training(self.model)

        # lora setup
        if efficient_finetuning_method == 'lora':
            config = LoraConfig(
                target_modules=lora_apply_layers, task_type="CAUSAL_LM", lora_dropout=lora_dropout
            )

            self.model = get_peft_model(self.model, config)

        # reft setup
        if efficient_finetuning_method == 'reft':
            config = ReftConfig(
                representations={
                    "component": reft_component,
                    "low_rank_dimension": reft_low_rank_dim,
                    "intervention": LoreftIntervention(
                        embed_dim=self.model.config.hidden_size,
                        low_rank_dimension=reft_low_rank_dim,
                    ),
                },
                low_rank_dimension=reft_low_rank_dim,
            )
            self.model = get_reft_model(self.model, config)

    def __optimizer(
        self, _load_in_4bit: bool, _load_in_8bit: bool, **unused_kwargs
    ) -> Callable[..., torch.optim.Optimizer]:
        if _load_in_4bit or _load_in_8bit:
            return "paged_adamw_8bit"
        return "adamw"

    def save_checkpoint(self, save_path: str):
        self.model.save_pretrained(save_path)

    def finetune(
        self,
        dataset: Dataset,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        data_collator: Callable[..., Any] = None,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.98),
        weight_decay: float = 1e-3,
        eps: float = 1e-8,
        epochs: int = 3,
    ):
        if not self.trainable:
            ValueError("currently not supporting finetuning with this model's configurations")
        training_args = TrainingArguments(
            "./out",
            per_device_train_batch_size=batch_size,
            optim=self.optimizer,
            bf16=(self.load_in_nbits <= 16),
            learning_rate=lr,
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_epsilon=eps,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if self.efficient_finetuning_method != "reft":
            trainer = HFTrainer(
                self.model,
                training_args,
                data_collator=data_collator,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
            )
        else:
            trainer = ReftTrainerForCausalLM(
                self.model,
                training_args,
                data_collator=data_collator,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
            )

        trainer.train()

    def generate(
        self,
        sentences: List[str],
        beam_size: int = 1,
        n_return_sequences: int = 1,
        top_k: int = 50,
        top_p: float = 1,
        temperature: float = 1,
    ):
        lengths = [len(sentence) for sentence in sentences]
        input_ids = self.tokenizer(sentences, padding=True, return_tensors='pt').input_ids
        gen_config = GenerationConfig(
            beam_size=beam_size,
            n_return_sequences=n_return_sequences,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        out_tokens = self.model.generate(input_ids, gen_config)
        out_sentences = self.tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
        return [out_sentences[i][length:] for i, length in enumerate(lengths)]
