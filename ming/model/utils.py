import torch 
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from peft.utils import _get_submodules
# from peft.tuners.lora import mark_only_lora_as_trainable
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union, Tuple
from safetensors import safe_open
import warnings

local_rank = None

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def mark_only_lora_as_trainable(model: nn.Module, bias) -> None:
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False
        if "lora" in n and "base" in n:
            p.requires_grad = False

    if bias == "none":
        return

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

def mark_only_orthlora_as_trainable(model: nn.Module, bias) -> None:
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False
        elif "lora" in n and "base" in n:
            p.requires_grad = False
        elif "lora" in n and "orth" in n:
            p.requires_grad = True

    if bias == "none":
        return

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

def check_target_module_exists(lora_config, key, target_modules):
    target_module_found = any(key.endswith(module_name) for module_name in target_modules)
    return target_module_found

def create_mixoflora_module(lora_config, target, model_args, add_bias=True):
    in_features, out_features = target.in_features, target.out_features
    # print(lora_config)
    new_module = MoLoRALinear(in_features, out_features, model_args.num_experts, model_args.num_experts_per_token,
                              r=lora_config.r,
                              lora_alpha=lora_config.lora_alpha,
                              lora_dropout=lora_config.lora_dropout,
                              use_rslora=lora_config.use_rslora,
                              enable_router_loss=model_args.enable_router_loss,
                              expert_sampling=True if model_args.expert_selection == "sampling" else False,
                              use_lbl_loss=model_args.use_lbl_loss,
                              share_expert=model_args.share_expert,
                              num_share_experts=model_args.num_share_experts,
                              bias=add_bias)
    return new_module

def create_orthlora_module(lora_config, target, model_args, add_bias=True):
    in_features, out_features = target.in_features, target.out_features
    new_module = OrthLoRALinear(in_features, out_features, r=lora_config.r,
                                lora_alpha=lora_config.lora_alpha,
                                lora_dropout=lora_config.lora_dropout,
                                use_rslora=lora_config.use_rslora,
                                bias=add_bias)
    return new_module

def get_orthlora_model(model, model_args, lora_config, decoder_type=Qwen2DecoderLayer, lora_name_or_path=None, inference_mode=False):
    
    # first merge lora attention weights from the lora_name_or_path/adapter_model.safetensors
    # attn_lora_params = {}
    # rank0_print(lora_name_or_path)
    # with safe_open(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), framework="pt", device=0) as f:
    #     for k in f.keys():
    #         attn_lora_params[k] = f.get_tensor(k)
    # initial_base_params = {}
    # for k in attn_lora_params:
    #     # find self_attn.k_proj.lora_A.weight 
    #     if "self_attn" in k and ("lora_A" in k or "lora_B" in k):
    #         initial_base_params[k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")] = attn_lora_params[k]
    # incompatible_keys = model.load_state_dict(initial_base_params, strict=False)
    # rank0_print(incompatible_keys)
    # model.merge_and_unload()

    # find linear modules with "switch" in their attributes
    key_list = [key for key, _ in model.named_modules()]
    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), decoder_type) and "mlp" in name:
                names = name.split(".")
                target_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_module_names = list(target_module_names)
    
    for key in key_list:
        if not check_target_module_exists(lora_config, key, target_module_names):
            continue
            
        parent, target, target_name = _get_submodules(model, key)
        # print(parent, target_name)
        if hasattr(target, "bias"):
            if target.bias is not None:
                add_bias = True 
            else: 
                add_bias = False
        else:
            add_bias = False
        # new_module = create_mixoflora_module(lora_config, target, num_experts, num_experts_per_token, enable_router_loss, True if expert_selection == "sampling" else False, use_lbl_loss, add_bias=add_bias)
        new_module = create_orthlora_module(lora_config, target, model_args, add_bias=add_bias)
        setattr(parent, target_name, new_module)
        new_module.weight = target.weight 
        if hasattr(target, "bias"):
            if target.bias is not None:
                new_module.bias = target.bias

        new_module.to(target.weight.device)
        
        if getattr(target, "state", None) is not None:
            new_module.state = target.state
            new_module.to(target.weight.device)
        
        del target
    
    if lora_name_or_path is not None and os.path.exists(lora_name_or_path):
        print(f"Initializing base LoRA weights from share_experts in {os.path.join(lora_name_or_path, 'non_lora_trainables.bin')}")
        # shared_ffn_lora_params = {}
        # non_lora_trainables = torch.load(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), map_location='cpu')
        # with safe_open(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), framework="pt", device=0) as f:
        #     for k in f.keys():
        #         lora_params[k] = f.get_tensor(k)
        shared_ffn_lora_params = torch.load(os.path.join(lora_name_or_path, 'non_lora_trainables.bin'), map_location='cpu')
        shared_ffn_lora_params = {(k[11:] if k.startswith('base_model.') else k): v for k, v in shared_ffn_lora_params.items()}
        if any(k.startswith('model.model.') for k in shared_ffn_lora_params):
            shared_ffn_lora_params = {(k[6:] if k.startswith('model.') else k): v for k, v in shared_ffn_lora_params.items()}


        initial_base_params = {}
        for k in shared_ffn_lora_params:
            if "mlp" in k and "lora" in k and "share_experts" in k:
                initial_base_params[k.replace("share_experts.lora_A", "base.lora_A").replace(
                    "share_experts.lora_B", "base.lora_B"
                )] = shared_ffn_lora_params[k]
            elif "self_attn" in k and "lora" in k:
                initial_base_params[k] = shared_ffn_lora_params[k]
            elif "mlp" in k and ".experts." in k:
                continue 
            elif "switch" in k:
                continue
            else:
                rank0_print(k)
                exit(-1)
        # for k in attn_lora_params:
        #     # find self_attn.k_proj.lora_A.weight 
        #     if "self_attn" in k and ("lora_A" in k or "lora_B" in k):
        #         initial_base_params[k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")] = attn_lora_params[k]

            # else:
            #     initial_base_params[k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")] = lora_params[k]
        # print("Loading Parameters: ", initial_molora_params.keys())
        # for k, v in initial_base_params.items():
        #     print(k)
        # for k, v in model.state_dict().items():
        #     print(k)
        incompatible_keys = model.load_state_dict(initial_base_params, strict=False)
        # print(incompatible_keys)
        # print(incompatible_keys)
    else:
        print("No initialization for Base!")

    mark_only_lora_as_trainable(model, getattr(lora_config, "bias", "none"))
    if inference_mode:
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
    else:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_parameters()
    
    return model


def merge_and_unload(model):
    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue
        if isinstance(target, LoRALayer):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            target.merge()
            replace_module(parent, target_name, new_module, target)



    return model

def replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias"):
        if old_module.bias is not None:
            new_module.bias = old_module.bias

    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            module.to(old_module.weight.device)
            
def get_mixoflora_model(model, model_args, lora_config, decoder_type=Qwen2DecoderLayer, lora_name_or_path=None, inference_mode=False):
    # find linear modules with "switch" in their attributes
    key_list = [key for key, _ in model.named_modules()]
    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), decoder_type) and "mlp" in name:
                names = name.split(".")
                target_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_module_names = list(target_module_names)
    
    for key in key_list:
        if not check_target_module_exists(lora_config, key, target_module_names):
            continue
            
        parent, target, target_name = _get_submodules(model, key)
        # print(parent, target_name)
        if hasattr(target, "bias"):
            if target.bias is not None:
                add_bias = True 
            else: 
                add_bias = False
        else:
            add_bias = False
        # new_module = create_mixoflora_module(lora_config, target, num_experts, num_experts_per_token, enable_router_loss, True if expert_selection == "sampling" else False, use_lbl_loss, add_bias=add_bias)
        new_module = create_mixoflora_module(lora_config, target, model_args, add_bias=add_bias)
        # setattr(parent, target_name, new_module)
        # new_module.weight = target.weight 
        # if hasattr(target, "bias"):
        #     if target.bias is not None:
        #         new_module.bias = target.bias

        # new_module.to(target.weight.device)
        
        # if getattr(target, "state", None) is not None:
        #     new_module.state = target.state
        #     new_module.to(target.weight.device)
        replace_module(parent, target_name, new_module, target)
        
        del target
    
    # print(lora_name_or_path)
    if lora_name_or_path is not None and os.path.exists(lora_name_or_path):
        print(f"Initializing MoLoRA from {os.path.join(lora_name_or_path, 'adapter_model.safetensors')}")
        lora_params = {}
        # non_lora_trainables = torch.load(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), map_location='cpu')
        with safe_open(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), framework="pt", device=0) as f:
            for k in f.keys():
                lora_params[k] = f.get_tensor(k)
            # lora_params = {(k[11:] if k.startswith('base_model.') else k): v for k, v in lora_params.items()}
            # if any(k.startswith('model.model.') for k in lora_params):
            #     lora_params = {(k[6:] if k.startswith('model.') else k): v for k, v in lora_params.items()}
            # print("LoRA Parameters: ", lora_params.keys())

        initial_molora_params = {}
        for k in lora_params:
            if "mlp" in k and "lora" in k:
                for i in range(model_args.num_experts):
                    initial_molora_params[k.replace("lora_A", f"experts.{i}.lora_A_{i}").replace("lora_B", f"experts.{i}.lora_B_{i}")] = lora_params[k]
            else:
                initial_molora_params[k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")] = lora_params[k]
        # print("Loading Parameters: ", initial_molora_params.keys())
        incompatible_keys = model.load_state_dict(initial_molora_params, strict=False)
        # print(incompatible_keys)
    else:
        print("No initialization for MoLoRA!")

    mark_only_lora_as_trainable(model, getattr(lora_config, "bias", "none"))
    if inference_mode:
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
    else:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_parameters()
    
    return model


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRAModule, self).__init__()
        self.lora_a = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_b = nn.Parameter(torch.zeros((out_features, r)))
        self.reset_parameters()

    def forward(self):
        return self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

class OrthLoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        use_rslora: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.fan_in_fan_out = fan_in_fan_out

        self.use_rslora = use_rslora
        nn.Linear.reset_parameters(self)
        
        if r > 0:
            self.base = nn.ModuleDict({"lora_A": nn.Linear(in_features, 2 * r, False, dtype=torch.float32), "lora_B": nn.Linear(2 * r, out_features, False, dtype=torch.float32)})
            self.orth = nn.ModuleDict({"lora_A": nn.Linear(in_features, 2 * r, False, dtype=torch.float32), "lora_B": nn.Linear(2 * r, out_features, False, dtype=torch.float32)})
            self.scaling = self.lora_alpha / (math.sqrt(self.r) if self.use_rslora else self.r)
            self.weight.requires_grad = False
            
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        
        if init_lora_weights:
            self.reset_lora_parameters()
        self.to(self.weight.device)
            
    def reset_lora_parameters(self):
        # nn.Linear.reset_parameters(self)
    
        if hasattr(self, 'base'):
            nn.init.kaiming_uniform_(self.base[f'lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.base['lora_B'].weight)
        
        if hasattr(self, 'orth'):
            nn.init.kaiming_uniform_(self.orth[f'lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.orth[f'lora_B'].weight)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            x = x.to(self.base['lora_A'].weight.dtype)
            x = self.lora_dropout(x)
            result += (
                self.base['lora_B'](
                    self.base['lora_A'](x)
                )
                * self.scaling
            )
            
            # modified 
            result += (
                self.orth['lora_B'](
                    self.orth['lora_A'](x)
                )
                * self.scaling
            )

        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
        result = result.to(previous_dtype)
        
        return result
    
    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return 
        if self.r > 0:
            self.weight.data += (
                transpose(
                    self.base['lora_B'].weight @ self.base['lora_A'].weight,
                    self.fan_in_fan_out
                ) * self.scaling + \
                    transpose(
                        self.orth['lora_B'].weight @ self.orth['lora_A'].weight,
                        self.fan_in_fan_out
                    ) * self.scaling 
            )
            self.merged = True
        

class MoLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        enable_router_loss: bool = False,
        use_lbl_loss: bool = False,
        share_expert: bool = False,
        num_share_experts: int = 1, 
        expert_sampling: bool = False,
        use_rslora: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # moe parameters
        self.num_experts = num_experts 
        self.num_experts_per_token = num_experts_per_token
        self.share_expert = share_expert
        self.expert_sampling = expert_sampling
        self.use_rslora = use_rslora

        self.enable_router_loss = enable_router_loss
        if num_experts > 1:
            self.switch = nn.Linear(in_features, num_experts)
        self.use_lbl_loss = use_lbl_loss    
        
        # Actual trainable parameters
        if r > 0:
            self.experts = nn.ModuleList([
                nn.ModuleDict({"lora_A_{}".format(i): nn.Linear(in_features, r, False, dtype=torch.float32),
                               "lora_B_{}".format(i): nn.Linear(r, out_features, False, dtype=torch.float32)})
            for i in range(num_experts)])

            self.scaling = self.lora_alpha / (math.sqrt(self.r) if self.use_rslora else self.r)
            self.weight.requires_grad = False

            if self.share_expert:
                self.share_experts = nn.ModuleDict({"lora_A": nn.Linear(in_features, r*num_share_experts, False, dtype=torch.float32),
                        "lora_B": nn.Linear(r*num_share_experts, out_features, False, dtype=torch.float32)})
        
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
    
        if hasattr(self, 'experts'):
            for idx, expert in enumerate(self.experts):
                nn.init.kaiming_uniform_(expert[f'lora_A_{idx}'].weight, a=math.sqrt(5))
                nn.init.zeros_(expert[f'lora_B_{idx}'].weight)
        
        if hasattr(self, 'share_experts'):
            nn.init.kaiming_uniform_(self.share_experts[f'lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.share_experts[f'lora_B'].weight)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)


    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.use_lbl_loss:
                moe_result, lbl_loss = self.molora_helper2(x)
                result += moe_result
                return result, lbl_loss
            elif self.enable_router_loss:
                moe_result, logit_sum = self.molora_helper2(x) if self.training else self.molora_helper(x)
                result += moe_result
                return result, logit_sum
            else:
                moe_result = self.molora_helper2(x) if self.training else self.molora_helper(x)
                result += moe_result
                return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    
    def molora_helper2(self, x: torch.Tensor):
        if self.num_experts <= 1:
            expert_output = self.experts[0]['lora_B_0'](
                self.experts[0]['lora_A_0'](self.lora_dropout(x))
            ) * self.scaling
            return expert_output
            
        previous_dtype = x.dtype 
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)
        x = x.to(self.experts[0]['lora_A_0'].weight.dtype)
        if self.share_expert:
            share_result = self.share_experts[f'lora_B'](self.share_experts[f'lora_A'](x)) * self.scaling # [bs * N, out_features]

        gate_logits = self.switch(x)  # [bs * N, expert]
        # if self.share_expert:
        #     gate_logits[:, 0] = -1e9
        #     temp_results = torch.stack([expert[f'lora_B_{i}'](expert[f'lora_A_{i}'](x)) * self.scaling for i, expert in enumerate(self.experts[1:])], dim=0)  # [expert, bs * N, out_features]
        # else:
        temp_results = torch.stack([expert[f'lora_B_{i}'](expert[f'lora_A_{i}'](x)) * self.scaling for i, expert in enumerate(self.experts)], dim=0)  # [expert, bs * N, out_features]
        temp_results = temp_results.transpose(0, 1)  # [bs * N, expert, out_features]

        if self.expert_sampling:
            selected_experts = torch.multinomial(F.softmax(gate_logits, dim=-1), self.num_experts_per_token, replacement=False)
            weights = torch.gather(gate_logits, 1, selected_experts)
            if self.enable_router_loss:
                gate_lprobs = torch.log_softmax(gate_logits, dim=-1)
                logit_sum = torch.gather(gate_lprobs, 1, selected_experts).sum(dim=-1)   
        else:
            gate_probs = gate_logits.softmax(-1)
            weights, selected_experts = torch.topk(gate_probs, self.num_experts_per_token)
            if self.use_lbl_loss:
                selected_num_per_experts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts) # self.num_experts
                fraction_per_expert = selected_num_per_experts / (batch_size * N) # self.num_experts
                prob_per_expert = gate_probs.mean(0)
                load_balancing_loss = (fraction_per_expert * prob_per_expert).sum() * self.num_experts # self.num_experts

            weights /= weights.sum(dim=-1, keepdim=True)

        selected_results = temp_results.gather(1, selected_experts.unsqueeze(-1).expand(-1, -1, self.out_features))  # [bs * N, select_expert, out_features]
        assert selected_results.shape == (batch_size * N, self.num_experts_per_token, self.out_features)
        if self.share_expert:
            weights = torch.cat([weights, torch.ones(weights.shape[0], 1).to(weights)], dim=-1)
            selected_results = torch.cat([
                selected_results,
                share_result.unsqueeze(1)
            ], dim=1)

        # weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        results = torch.einsum("be, bef -> bf", weights, selected_results)
        results = results.contiguous().view(batch_size, N, -1)
        results = results.to(previous_dtype)
        if self.enable_router_loss:
            return results, logit_sum
        elif self.use_lbl_loss:
            return results, load_balancing_loss
        else:
            return results
        
    def molora_helper(self, x: torch.Tensor):
        # debug:
        if self.num_experts <= 1:
            expert_output = self.experts[0]['lora_B_0'](
                self.experts[0]['lora_A_0'](self.lora_dropout(x))
            ) * self.scaling
            return expert_output
        batch_size, N, d = x.shape 
        previous_dtype = x.dtype
        x = x.contiguous().view(-1, d)       
        gate_logits = self.switch(x)  # [bs * N, expert]
        gate_probs = gate_logits.softmax(-1)

        if self.expert_sampling:
            selected_experts = torch.multinomial(gate_logits.exp(), self.num_experts_per_token, replacement=False)
            weights = torch.gather(gate_logits, 1, selected_experts)
        else:
            weights, selected_experts = torch.topk(gate_probs, self.num_experts_per_token)
            weights /= weights.sum(-1, keepdim=True)
        # weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        x = x.to(self.experts[0]['lora_A_0'].weight.dtype)
        results = torch.zeros((batch_size * N, self.out_features)).to(x) # bs*N, d
        load_balancing_loss = 0

        if batch_size == 1:
            assert selected_experts.shape[0] == 1
            selected_experts = selected_experts.flatten()
            weights = weights.flatten()
            for idx, expert_idx in enumerate(selected_experts):
                results += weights[idx] * self.experts[expert_idx]['lora_B_{}'.format(expert_idx)](
                    self.experts[expert_idx]['lora_A_{}'.format(expert_idx)](self.lora_dropout(x))
                ) * self.scaling
        else:
            for i, expert in enumerate(self.experts):
                batch_idx, nth_expert = torch.where(selected_experts == i) 
                expert_output = expert['lora_B_{}'.format(i)](
                    expert['lora_A_{}'.format(i)](self.lora_dropout(x[batch_idx]))
                ) * self.scaling # if i < self.num_experts else expert(self.lora_dropout(x[batch_idx]))
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_output
        
        results = results.contiguous().view(batch_size, N, self.out_features)
        results = results.to(previous_dtype)
        if self.share_expert:
            share_results = self.share_experts['lora_B'](
                self.share_experts['lora_A'](self.lora_dropout(x))
            ) * self.scaling
            results += share_results.contiguous().view(batch_size, N, self.out_features)

        if self.use_lbl_loss:
            return results, load_balancing_loss
        elif self.enable_router_loss:
            return_logit_sum = torch.zeros((batch_size * N, 1)).to(x)
            return results, return_logit_sum
        else:
            return results

def _select_loss(loss: torch.Tensor, inference_path, soft_select=False, length=-1):
    bs = loss.shape[0] // inference_path
    N =inference_path
    reshaped_loss = loss.view(N, bs)
    
    if soft_select:
        output_tensor = torch.mean(reshaped_loss, dim=0)
    else:
        # loss is a [bs * N,] tensor
        # 计算每组中N个元素的最小值，结果形状将是[1, bs, ...]
        # 使用min函数并指定dim=0来沿第一个维度（N的维度）进行计算
        output_tensor, minimum_index = torch.min(reshaped_loss, dim=0)
    output_tensor = output_tensor.sum(dim=0) / (length.sum())
    return output_tensor, minimum_index
        
def get_absolute_loss(logit_sum: Tuple[torch.Tensor], labels: torch.Tensor, minimum_path_index: torch.Tensor, inference_path: int, valid_length: torch.Tensor):
    L = len(logit_sum)
    bs = labels.shape[0] // inference_path
    logit_sum = torch.stack(logit_sum, dim=0).view(L, inference_path, -1, labels.shape[1])  # [L, N, bs, S]
    # use minimum_path_index select the logit_sum that infers the minimum loss
    minimum_logit_sum = logit_sum[:, minimum_path_index, torch.arange(logit_sum.shape[2]), :]  # [L, bs, S]
    # use labels to mask router_logit
    label_mask = labels[:bs, 1:].contiguous().ne(-100).float()
    router_logit = torch.sum(minimum_logit_sum[..., 1:], dim=0)  # [bs, S+1]

    masked_router_logit = router_logit * label_mask
    masked_router_logit_mean = masked_router_logit.sum() / (valid_length.sum())
    router_loss = -masked_router_logit_mean
    return router_loss 

def get_relative_loss(logit_sum: Tuple[torch.Tensor], labels: torch.Tensor, minimum_path_index: torch.Tensor, inference_path: int, valid_length: torch.Tensor):
    L = len(logit_sum)
    bs = labels.shape[0] // inference_path
    logit_sum = torch.stack(logit_sum, dim=0).view(L, 3, inference_path, -1, labels.shape[1])  # [L, 3, N, bs, S]
    # use minimum_path_index select the logit_sum that infers the minimum loss
    minimum_logit_sum = logit_sum[:, :, minimum_path_index, torch.arange(logit_sum.shape[3]), 1:]  # [L, 3, bs, S]
    # use labels to mask router_logit
    label_mask = labels[:bs, 1:].contiguous().ne(-100).float()  # [bs, S]
    masked_minimum_logit = minimum_logit_sum * label_mask  # [L, 3, bs, S]


    one_hot_minimum = torch.nn.functional.one_hot(minimum_path_index, num_classes=inference_path)  # [bs, N]
    one_hot_minimum = one_hot_minimum.transpose(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, bs, 1] to match [L, 3, N, bs, seq_length]
    mask = one_hot_minimum == 0  # Invert the mask: True where it's NOT the minimum

    # 应用掩码并重组数组以去除最小值对应的维度
    other_logits = logit_sum[mask.expand_as(logit_sum)].view(L, 3, inference_path-1, bs, -1)
    assert other_logits.shape[2] == 1
    other_logits = other_logits[..., 1:].squeeze(2)  # [L, 3, bs, S]
    # expand label mask to match other_logits's shape
    masked_other_logit = other_logits * label_mask
    contrastive_logit = torch.relu(masked_other_logit - masked_minimum_logit)
    router_loss = (contrastive_logit.sum() / (valid_length.sum() * 3 * L))

    return router_loss

def get_softmax_loss(logit_sum: Tuple[torch.Tensor], labels: torch.Tensor, minimum_path_index: torch.Tensor, inference_path: int, valid_length: torch.Tensor):
    L = len(logit_sum)
    bs = labels.shape[0] // inference_path
    logit_sum = torch.stack(logit_sum, dim=0).view(L, 3, inference_path, -1, labels.shape[1])  # [L, 3, N, bs, S]
    logit_sum_logsoftmax = F.log_softmax(logit_sum, dim=2)

    minimum_logit_sum = logit_sum_logsoftmax[:, :, minimum_path_index, torch.arange(logit_sum.shape[3]), :]  # [L, 3, bs, S]
    label_mask = labels[:bs, 1:].contiguous().ne(-100).float() # [bs, S]
    router_logit = torch.sum(minimum_logit_sum[..., 1:], dim=0)  # [3, bs, S]

    masked_router_logit = router_logit * label_mask # [3, bs, S]
    masked_router_logit_mean = masked_router_logit.sum() / (valid_length.sum() * 3 * L)
    router_loss = - masked_router_logit_mean
    return router_loss

def get_contrastive_loss(logit_sum: Tuple[torch.Tensor], labels: torch.Tensor, minimum_path_index: torch.Tensor, inference_path: int, valid_length: torch.Tensor):
    L = len(logit_sum)
    bs = labels.shape[0] // inference_path
    logit_sum = torch.stack(logit_sum, dim=0).view(L, 3, inference_path, -1, labels.shape[1])  # [L, 3, N, bs, S]
    logit_sum_logsoftmax = F.log_softmax(logit_sum, dim=2)

    minimum_logit_sum = logit_sum_logsoftmax[:, :, minimum_path_index, torch.arange(logit_sum.shape[3]), :]  # [L, 3, bs, S]
    label_mask = labels[:bs, 1:].contiguous().ne(-100).float() # [bs, S]
    router_logit = torch.sum(minimum_logit_sum[..., 1:], dim=0)  # [3, bs, S]

    masked_router_logit = router_logit * label_mask # [3, bs, S]
    masked_router_logit_mean = masked_router_logit.sum() / (valid_length.sum() * 3 * L)
    router_loss = - masked_router_logit_mean
    return router_loss

ROUTER_LOSS_MAPPING = {
    "absolute": get_absolute_loss,
    "relative": get_relative_loss,
    "softmax": get_softmax_loss,
    "contrastive_loss": get_contrastive_loss,
}

def multiple_path_forward(self, input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        enable_router_loss: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

    inference_path = 1 if not hasattr(self.config, "inference_path") else self.config.inference_path
    soft_select = False if not hasattr(self.config, "soft_select") else self.config.soft_select
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    enable_router_loss = enable_router_loss if enable_router_loss is not None else self.config.enable_router_loss
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        enable_router_loss=enable_router_loss
    )
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        if inference_path <= 1:
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            bs = logits.shape[0] // inference_path
            logit_sum = outputs[-1]  # ([3 * N * bs * seq_length], ) * L
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss_per_batch = loss_fct(shift_logits, shift_labels)  # [N * bs * seq_length, ]

            loss_per_batch = loss_per_batch.view(-1, labels.shape[1] - 1).sum(dim=-1)  # [bs * N]
            length = labels[:bs, 1:].contiguous().ne(-100).float().sum(-1)
            same_rate = loss_per_batch[:bs].eq(loss_per_batch[bs:]).float().mean()

            loss, minimum_path_index = _select_loss(loss_per_batch, inference_path, soft_select, length)
            if self.config.enable_router_loss:
                router_loss = ROUTER_LOSS_MAPPING[self.config.router_loss_mode](logit_sum, labels, minimum_path_index, inference_path, length)
                loss += self.config.router_loss_coeff * router_loss

                print(f"Total Loss: {loss}, NLL Loss: {loss - self.config.router_loss_coeff * router_loss}, Router Loss: {router_loss}, weight: {self.config.router_loss_coeff} Same rate: {same_rate}")

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def lbl_loss_forward(self, input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        enable_router_loss: Optional[bool] = None,
        use_lbl_loss: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

    inference_path = 1 if not hasattr(self.config, "inference_path") else self.config.inference_path
    soft_select = False if not hasattr(self.config, "soft_select") else self.config.soft_select
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    enable_router_loss = enable_router_loss if enable_router_loss is not None else self.config.enable_router_loss
    use_lbl_loss = use_lbl_loss if use_lbl_loss is not None else self.config.use_lbl_loss
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        enable_router_loss=enable_router_loss,
        use_lbl_loss=True
    )
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        if inference_path <= 1:
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if use_lbl_loss:
                lbl_loss = outputs[-1]
                lbl_loss = sum(lbl_loss) / len(lbl_loss)
                loss += self.config.lbl_loss_coeff * lbl_loss
                print(f"Total Loss: {loss}, NLL Loss: {loss - self.config.lbl_loss_coeff * lbl_loss}, Router Loss: {lbl_loss}, weight: {self.config.lbl_loss_coeff}")
        else:
            bs = logits.shape[0] // inference_path
            logit_sum = outputs[-1]  # ([3 * N * bs * seq_length], ) * L
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss_per_batch = loss_fct(shift_logits, shift_labels)  # [N * bs * seq_length, ]

            loss_per_batch = loss_per_batch.view(-1, labels.shape[1] - 1).sum(dim=-1)  # [bs * N]
            length = labels[:bs, 1:].contiguous().ne(-100).float().sum(-1)
            same_rate = loss_per_batch[:bs].eq(loss_per_batch[bs:]).float().mean()

            loss, minimum_path_index = _select_loss(loss_per_batch, inference_path, soft_select, length)
            if enable_router_loss:
                router_loss = ROUTER_LOSS_MAPPING[self.config.router_loss_mode](logit_sum, labels, minimum_path_index, inference_path, length)
                loss += self.config.router_loss_coeff * router_loss

                print(f"Total Loss: {loss}, NLL Loss: {loss - self.config.router_loss_coeff * router_loss}, Router Loss: {router_loss}, weight: {self.config.router_loss_coeff} Same rate: {same_rate}")

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )