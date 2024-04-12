import os
import warnings
import shutil
from copy import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from ming.model.utils import get_mixoflora_model
import json

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, use_logit_bias=False, only_load=None, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if model_base is not None:
        # PEFT model
        from peft import PeftModel, PeftConfig
        tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_base, trust_remote_code=True, low_cpu_mem_usage=True, **kwargs)
        print(f"Loading LoRA weights from {model_path}")
        lora_config = PeftConfig.from_pretrained(model_path)
        if only_load == "attn":
            lora_config.target_modules = {m for m in lora_config.target_modules if m not in ["up_proj", "down_proj", "gate_proj"]}

        elif only_load == "ffn":
            lora_config.target_modules = {m for m in lora_config.target_modules if m in ["up_proj", "down_proj", "gate_proj"]}
        
        model = PeftModel.from_pretrained(model, model_path, config=lora_config)
        print(f"Merging weights")
        model = model.merge_and_unload()
        print('Convert to FP16...')
        model.to(torch.float16)

        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base, add_prefix_space=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
        
    return tokenizer, model, context_len, tokenizer_with_prefix_space


def load_molora_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, use_logit_bias=False, only_load=None, device_map="auto", device="cuda", expert_selection=None):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

        # Load language model
    if model_base is not None:
        # PEFT model
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        
        if expert_selection:
            lora_cfg_pretrained.expert_selection = expert_selection
        if not hasattr(lora_cfg_pretrained, "num_experts"):
            lora_cfg_pretrained.num_experts = 4
            lora_cfg_pretrained.num_experts_per_token = 2
            lora_cfg_pretrained.share_expert = False
            lora_cfg_pretrained.expert_selection = "top_k"
        if getattr(lora_cfg_pretrained, "use_rslora", None) is None:
            setattr(lora_cfg_pretrained, "use_rslora", False)
            
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            lora_specific_pretrained = json.load(f)
            # merge lora_specific_pretrained to lora_cfg_pretrained, which is a transformer config class
            lora_cfg_pretrained.r = lora_specific_pretrained['r']
            lora_cfg_pretrained.lora_alpha = lora_specific_pretrained["lora_alpha"]
            lora_cfg_pretrained.lora_dropout = lora_specific_pretrained["lora_dropout"]
            lora_cfg_pretrained.bias = lora_specific_pretrained['bias']
        
        print(lora_specific_pretrained)
        if only_load != "ffn":
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            from peft import PeftModel
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        
        # all_mlp_weights = [y for x, y in model.state_dict().items() if "gate_proj.weight" in x or "down_proj.weight" in x or "up_proj.weight" in x]
        if only_load != "attn":
            model = get_mixoflora_model(model, lora_cfg_pretrained.num_experts,
                                            lora_cfg_pretrained.num_experts_per_token,
                                            expert_selection=lora_cfg_pretrained.expert_selection,
                                            lora_config=lora_cfg_pretrained,
                                            use_logit_sum=False,
                                            inference_mode=True)
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            incompatible_keys = model.load_state_dict(non_lora_trainables, strict=False)
            # print(incompatible_keys)
            
        print('Convert to FP16...')
        model.to(torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
    else:
        tokenizer_with_prefix_space = None
        
    return tokenizer, model, context_len, tokenizer_with_prefix_space