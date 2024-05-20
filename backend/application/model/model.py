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
        lora_r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        reft_component: int | str = None,
        reft_low_rank_dim: int = None,
    ):
        """
        Wrapper for Llama 3 model, including training and inference.

        Params:
        - `model`: the Llama 3 model size. Defaults to the `8B-Instruct` version.
        - `custom_checkpoint_path`: the model (or adapter if the model was finetuned with LoRa) checkpoint saved with `Llama3Model.save_checkpoint`.
        - `load_in_n_bits`: Specifies the fp/quantization type. `32` uses fp32, `16` uses bf16, `8` uses llm.int8 quantization, `4` uses nf4 quantization from QLoRa.
        - `efficient_finetuning_method`: The efficient finetuning method for the model. Currently only supports LoRa (QLoRa if `load_in_n_bits=4`).
            `reft` (Representation Fine-tuning) is there to show that I can do it if allowed to, however `pyreft` does not have support for Llama 3 tokenizer yet,
            so it is a work in progress.
        - `lora_apply_layers`: Layers to be applied with LoRa, if specified.
        - `lora_dropout`: The dropout rate used with LoRa if specified.
        - `lora_r`: The rank used for LoRa if specified.
        - `lora_alpha`: LoRa α, used if LoRa training is specified.
        - `reft_component`: The layer and layer component to use ReFT.
        - `reft_low_rank_dim`: The rank used for ReFT.
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
                target_modules=lora_apply_layers,
                task_type="CAUSAL_LM",
                lora_dropout=lora_dropout,
                r=lora_r,
                lora_alpha=lora_alpha,
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
        """
        Finetune the model with configurations specified.

        Params:
        - `dataset`: a `torch.utils.data.Dataset` where indexing it returns an element as a dict with the keys:
            - `input_ids` and `labels`: the tokens extracted from the Llama 3 tokenizer, padded and truncated to a fixed length
            - `attention_mask`: the attention mask generated with the Llama 3 tokenizer.
        - `batch_size`: the training batch size.
        - `gradient_accumulation_steps`: how many steps to accumulate gradients before performing a backward pass
        - `data_collator`: preferably of type `transformers.DataCollator`, the data collator used for the Hugging Face trainer.
        - `lr`, `betas`, `weight_decay`, `eps`: The usual parameters for the AdamW optimizer.
        - `epochs`: number of training epochs.
        """
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
        sentences: List[List[dict]] | List[dict],
        beam_size: int = 1,
        n_return_sequences: int = 1,
        top_k: int = 50,
        top_p: float = 1,
        temperature: float = 1,
    ):
        """
        Generate the next line of the conversation.

        Params:
        - `sentences`: List (or list of lists) of dicts specifying roles and messages of the conversation.
        - `beam_size`: Beam size for beam search
        - `n_return_sequences`: How many versions of the next line should be generated
        - `top_k`: specifies how many tokens with highest softmax probabilities to be sampled (Top-k sampling)
        - `top_p`: specifies the cummulative probability limit to choose tokens with highest softmax probabilities (Top-p/Nucleus sampling)
        - `temperature`: the temperature of the generation: lower = less creative, follows more to the training data; higher = more creative, but also higher rate of hallucination.
        """
        if isinstance(sentences[0], dict):
            sentences = [sentences]
        input_ids = self.tokenizer.apply_chat_template(
            sentences, add_generation_prompt=True, padding=True, return_tensors='pt'
        )
        lengths = input_ids.shape[-1]
        gen_config = GenerationConfig(
            beam_size=beam_size,
            n_return_sequences=n_return_sequences,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_length=1024,
        )
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        out_tokens = self.model.generate(
            input_ids, gen_config, eos_token_id=terminators, do_sample=True
        )
        out_sentences = self.tokenizer.batch_decode(
            out_tokens[:, lengths:], skip_special_tokens=True
        )

        return out_sentences
