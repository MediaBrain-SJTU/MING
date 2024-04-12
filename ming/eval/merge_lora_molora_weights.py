import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

import argparse
import os 
import torch
from ming.model.builder import load_molora_pretrained_model
from ming.utils import disable_torch_init, get_model_name_from_path
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
def convert_to_automodel(model_path, model_base, load_8bit=False, load_4bit=False, use_logit_bias=False, device_map="auto", device="cuda", save_path=None):
    disable_torch_init()
    # assert model_path contains adapter_model.bin and non_lora_trainables.bin two files
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, context_len, tokenizer_with_adapter = load_molora_pretrained_model(model_path, model_base, model_name)
    if save_path is None:
        return 

    os.makedirs(save_path)
    config = model.config 
    model: Qwen2ForCausalLM
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model save path")
    parser.add_argument("--model_base", type=str, help="the path of the base model that a peft model has")
    parser.add_argument("--save_path", type=str, help='the full model\'s save path')
    
    args = parser.parse_args()
    model_path = args.model_path
    model_base = args.model_base
    save_path = args.save_path
    torch.set_default_device("cuda")
    convert_to_automodel(model_path, model_base, save_path=save_path)