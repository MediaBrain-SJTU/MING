import os
import warnings
import shutil
from copy import copy
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from ming.model.utils import get_mixoflora_model, get_orthlora_model, merge_and_unload
import json


def load_pretrained_orth_model(model_path, model_base, lora_name_or_path, load_8bit=False, load_4bit=False, use_logit_bias=False, only_load=None, device_map="auto", device="cuda"):
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
        print(f"Loading Base and Orthogonal LoRA weights from {model_path}")
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        lora_config = PeftConfig.from_pretrained(model_path)
        model = get_orthlora_model(model, lora_cfg_pretrained, lora_config, lora_name_or_path=lora_name_or_path)
        orth_lora_weights = load_file(os.path.join(model_path, "model.safetensors"))
        orth_lora_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in orth_lora_weights.items()}
        if any(k.startswith('model.model.') for k in orth_lora_weights):
            orth_lora_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in orth_lora_weights.items()}
        
        # orth_lora_weights = {k : v for k, v in orth_lora_weights.items() if "orth" in k}
        
        model.load_state_dict(orth_lora_weights, strict=False)
        model = merge_and_unload(model)
        
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
        # if not hasattr(lora_cfg_pretrained, "num_experts"):
        lora_cfg_pretrained.num_experts = getattr(lora_cfg_pretrained, "num_experts", 4)
        lora_cfg_pretrained.num_experts_per_token = getattr(lora_cfg_pretrained, "num_experts_per_token", 2)
        lora_cfg_pretrained.share_expert = getattr(lora_cfg_pretrained, "share_expert", False)
        lora_cfg_pretrained.num_share_experts = getattr(lora_cfg_pretrained, "num_share_experts", 1)
        lora_cfg_pretrained.expert_selection = getattr(lora_cfg_pretrained, "expert_selection", "top_k")
        if getattr(lora_cfg_pretrained, "use_rslora", None) is None:
            setattr(lora_cfg_pretrained, "use_rslora", False)
        
        from peft import PeftModel, PeftConfig
        lora_config = PeftConfig.from_pretrained(model_path)
        
        print(lora_config)
        if only_load != "ffn":
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            from peft import PeftModel
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path, config=lora_config)

            print(f"Merging LoRA weights")
            model = model.merge_and_unload()

            if only_load == "share": 
                print(f"Loading Share Expert of the LoRA weights from {model_path}")

                lora_config.target_modules = set(["up_proj", "down_proj", "gate_proj"])
                lora_config.r = lora_config.r * lora_cfg_pretrained.num_share_experts

                model = PeftModel.from_pretrained(model, model_path, config=lora_config)
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                share_experts_params = {}
                for k, v in non_lora_trainables.items():
                    if "share_experts" in k and "lora" in k:
                        share_experts_params[k.replace('share_experts.', '').replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default')] = v

                incompatible_keys = model.load_state_dict(share_experts_params, strict=False)
                print(incompatible_keys)

                print(f"Merging Share Expert of MoLoRA weights")
                model = model.merge_and_unload()

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        
        # all_mlp_weights = [y for x, y in model.state_dict().items() if "gate_proj.weight" in x or "down_proj.weight" in x or "up_proj.weight" in x]
        if only_load != "attn" and only_load != "share":
            print(f"Loading MoLoRA weights from {model_path}")
            if only_load == "no_share":
                lora_cfg_pretrained.share_expert = False
                
            model = get_mixoflora_model(model, model_args=lora_cfg_pretrained, lora_config=lora_config)
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            incompatible_keys = model.load_state_dict(non_lora_trainables, strict=False)
            
        print('Convert to FP16...')
        model.to(torch.float16)

        # print(*model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    if model_base is not None:
        # lora case
        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
    else:
        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)

    return tokenizer, model, context_len, tokenizer_with_prefix_space