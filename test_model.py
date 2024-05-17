from backend.model.model import Llama3Model


def test_model():
    model = Llama3Model("8B-Instruct", load_in_n_bits=16)
    tok = model.tokenizer
    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    print(tok.decode(response, skip_special_tokens=True))
