import torch


## simple wrapper function
def load_model(model_path: str, dtype=torch.float16):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
