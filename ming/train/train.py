# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import jsonlines
import logging
import pathlib
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Sequence, List
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
# from ming.model.modeling_phi import PhiForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM
import torch
import warnings
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import Dataset
from ming.conversations import get_default_conv_template, SeparatorStyle
import pdb
from ming.model.utils import get_mixoflora_model, get_orthlora_model, multiple_path_forward, lbl_loss_forward
import warnings
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from ming.train.trainer import MINGTrainer
from ming.model import MoLoRAQwenForCausalLM, MoLoRAQwenMLP
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    
    # mix of lora arguments
    num_experts: Optional[int] = field(default=1)
    num_experts_per_token: Optional[int] = field(default=1)
    expert_selection: Optional[str] = field(default="top_k", metadata={"help": "top_k or sampling"})
    share_expert: Optional[bool] = field(default=False)
    num_share_experts: Optional[int] = field(default=1)
    enable_router_loss: Optional[bool] = field(default=False)
    router_loss_coeff: Optional[float] = field(default="0.")
    router_loss_mode : Optional[str] = field(default="contrastive", metadata={"help": "contrastive or softmax"})

    use_lbl_loss: Optional[bool] = field(default=False)
    lbl_loss_coeff: Optional[float] = field(default="0.")

@dataclass
class DataArguments:
    train_data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    val_data_path: str = field(default=None,
                           metadata={"help": "Path to the validation data."})
    prompt_type: str = field(default="qwen",
                           metadata={"help": "prompt type"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_attn_enable: bool = False
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_use_rs: bool = False
    
    # our method
    lora_name_or_path: str = None
    inference_path: Optional[int] = field(default=1)
    soft_select: Optional[bool] = field(default=False)
    
    # orthogonal share
    lamda_1: Optional[float] = field(default=0.5)
    lamda_2: Optional[float] = field(default=0.)
    
    # orthogonal twostage training
    use_orthogonal: Optional[bool] = field(default=False)
    

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k and "experts" not in k and t.requires_grad}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k or ("lora_" in k and "experts" in k)}
    # mix of lora parameters
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, wrap_projector=False, whether_wrap_ffn=True):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'switch']
    if wrap_projector:
        multimodal_keywords.remove("mm_projector")
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if not whether_wrap_ffn:
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), Qwen2DecoderLayer) and isinstance(module, cls) and "mlp" in name:
                continue
            
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    prompt_type,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    # max_len: int,
    # system_message: str = "You are a helpful assistant."
) -> Dict:
    conv = get_default_conv_template(prompt_type).copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    system_message = conv.system

    im_start = tokenizer("<|im_start|>")["input_ids"][-1]
    im_end = tokenizer("<|im_end|>")["input_ids"][-1]
    nl_tokens = tokenizer('\n', add_special_tokens=False).input_ids
    _system = tokenizer('system', add_special_tokens=False).input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    if "role" in sources[0][0]:
        role_key = "role"
    else:
        role_key = "from"

    for i, source in enumerate(sources):
        if roles[source[0][role_key]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message, add_special_tokens=False).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        # import pdb
        # pdb.set_trace()
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence[role_key]]
            _input_id = tokenizer(role, add_special_tokens=False).input_ids + nl_tokens + \
                tokenizer(sentence["value"], add_special_tokens=False).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role, add_special_tokens=False).input_ids) + \
                    _input_id[len(tokenizer(role, add_special_tokens=False).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(input_id))
        target += [IGNORE_TOKEN_ID] * (tokenizer.model_max_length - len(target))
        input_ids.append(input_id[:tokenizer.model_max_length])
        targets.append(target[:tokenizer.model_max_length])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 prompt_type: str):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(prompt_type, sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 prompt_type: str,):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type

        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        rank0_print("Loading total {} instances...".format(len(list_data_dict)))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess(self.prompt_type, [e["conversations"] for e in sources],
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             attention_mask=data_dict["attention_mask"][0])
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        # input_ids, labels = tuple([torch.tensor(instance[key]).long() for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = [], []
        for instance in instances:
            instance_len = instance["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1)
            input_ids.append(instance["input_ids"][:instance_len].long())
            labels.append(instance["labels"][:instance_len].long())

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class MultiplePathDataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    inference_path: int  = 1

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = [], []
        for instance in instances:
            instance_len = instance["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1)
            input_ids.append(instance["input_ids"][:instance_len].long())
            labels.append(instance["labels"][:instance_len].long())

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        
        input_ids, labels, attention_mask = self._repeat_input(input_ids, labels, attention_mask)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    
    def _repeat_input(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        if self.inference_path == 1:
            return input_ids, labels, attention_mask
        input_ids = input_ids.repeat(self.inference_path, 1)
        labels = labels.repeat(self.inference_path, 1)
        attention_mask = attention_mask.repeat(self.inference_path, 1) 
        
        return input_ids, labels, attention_mask


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                inference_path:int = 1) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.train_data_path,
                                prompt_type=data_args.prompt_type)
    eval_dataset = dataset_cls(tokenizer=tokenizer,
                                         data_path=data_args.val_data_path,
                                prompt_type=data_args.prompt_type) if data_args.val_data_path is not None else None
    
    if inference_path > 1:
        data_collator = MultiplePathDataCollatorForSupervisedDataset(
            tokenizer=tokenizer, inference_path=inference_path
        )
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # add new config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, trust_remote_code=True)
    config.enable_router_loss = model_args.enable_router_loss
    config.router_loss_coeff = model_args.router_loss_coeff
    config.router_loss_mode = model_args.router_loss_mode
    config.use_lbl_loss = model_args.use_lbl_loss
    config.lbl_loss_coeff = model_args.lbl_loss_coeff
    
    if model_args.num_experts > 1:
        config.num_experts = model_args.num_experts
        config.num_experts_per_token = model_args.num_experts_per_token
        config.expert_selection = model_args.expert_selection
        config.share_expert = model_args.share_expert
        config.num_share_experts = model_args.num_share_experts
        
    if training_args.inference_path > 1:
        print("redefine the model.forward as multiple_path_forward")
        config.inference_path = training_args.inference_path
        config.soft_select = training_args.soft_select
        
        # replace the qwen2 model's forward to current modified forward function
        MoLoRAQwenForCausalLM.forward = multiple_path_forward
    
    if model_args.use_lbl_loss:
        print("redefine the model.forward as lbl_loss_forward")
        MoLoRAQwenForCausalLM.forward = lbl_loss_forward

    model = MoLoRAQwenForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **bnb_model_from_pretrained_args)    
    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable or training_args.lora_attn_enable:
        from peft import LoraConfig, get_peft_model, PeftModel
        import peft 
        if peft.__version__ == "0.9.0":
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model, False, whether_wrap_ffn=True if (model_args.num_experts <= 1 and not training_args.lora_attn_enable) else False),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
                use_rslora=training_args.lora_use_rs
            )
        else:
            if training_args.lora_use_rs:
                warnings.warn("You set use_rslora as True when using an unsupported peft version; try `pip install peft --upgrade` to fix it.")
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model, False, whether_wrap_ffn=True if (model_args.num_experts <= 1 and not training_args.lora_attn_enable) else False),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        if training_args.use_orthogonal:
            pass
        else:
            model = get_peft_model(model, lora_config)
        
    if model_args.num_experts > 1:
        # get mix of lora model
        # if model_args.share_expert:
        #     warnings.warn("Not support expert sharing yet; back to non-sharing mode")
        model = get_mixoflora_model(model, model_args, lora_config=lora_config, lora_name_or_path=training_args.lora_name_or_path)
        rank0_print(model.config)
        rank0_print(model)
        training_args.molora = True 
    else:
        training_args.molora = False
    if training_args.use_orthogonal:
        rank0_print("Create orthogonal model...")
        model = get_orthlora_model(model, model_args, lora_config=lora_config, lora_name_or_path=training_args.lora_name_or_path)
        rank0_print("orthogonal model:", model)
        

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,
                                                           cache_dir=training_args.cache_dir,
                                                           model_max_length=training_args.model_max_length,
                                                           use_fast=False)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              inference_path=training_args.inference_path)
    # print(model)
    for n, p in model.named_parameters():
        if p.requires_grad:
            rank0_print(n)

    trainer = MINGTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable or training_args.lora_attn_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            lora_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()