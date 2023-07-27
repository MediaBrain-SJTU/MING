# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

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

from dataclasses import dataclass, field
import logging
import pathlib
import typing

import pdb
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import BottleneckConfig, get_peft_model
import transformers
from transformers import Trainer

from fastchat.train.train import (DataArguments, ModelArguments,
                                  TrainingArguments,
                                  make_supervised_data_module)

from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from lion_pytorch import Lion
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar,create_lr_scheduler_with_warmup

replace_llama_attn_with_flash_attn()

@dataclass
class AdapterArguments:
    bottleneck_size: int = 512
    non_linearity: str = 'tanh'
    adapter_dropout: float = 0.1
    use_parallel_adapter: bool = True
    use_adapterp: bool = False
    scaling: float = 1.0
    target_modules: typing.Dict = {"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"}
    bias: str = "none"

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == "none":
        to_return = {k: state_dict[k].cpu().clone().detach() for k in state_dict if "lora_" in k}
    elif bias == "all":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, AdapterArguments))
    (model_args, data_args, training_args,
     adapter_args) = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map="auto"
    )
    model.model_parallel = True

    adapter_config = BottleneckConfig(
        bottleneck_size=adapter_args.bottleneck_size,
        non_linearity=adapter_args.non_linearity,
        adapter_dropout=adapter_args.adapter_dropout,
        use_parallel_adapter=adapter_args.use_parallel_adapter,
        use_adapterp=adapter_args.use_adapterp,
        target_modules={"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
        scaling=adapter_args.scaling,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter_config)
    # print(model)
    # pdb.set_trace()
    for layer_i in range(len(model.base_model.model.model.layers)):
        device = model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.weight.device
        model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.adapter_down.half().to(device)
        model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.adapter_up.half().to(device)
        model.base_model.model.transformer.layers[layer_i].mlp.dense_4h_to_h.adapter_down.half().to(device)
        model.base_model.model.transformer.layers[layer_i].mlp.dense_4h_to_h.adapter_up.half().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters:{total_params:,} || training parameters:{total_trainable_params:,} || trainable%:{total_trainable_params/total_params * 100}')
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()


    if training_args.gradient_checkpointing:
        logging.warning("gradient checkpointing with lora makes requires_grad "
                        "incorrect and needs a monkey patch in Trainer or the "
                        "wrapped model's forward. ref: "
                        "https://github.com/lm-sys/FastChat/pull/138#issuecomment-1509172198")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    optimizer = Lion(model.parameters(), lr=training_args.learning_rate)
    # pdb.set_trace()
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      optimizers=(optimizer, None),
                      **data_module)
    # pdb.set_trace()
    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Save states. Weights might be a placeholder in zero3 and need a gather
    state_dict = get_peft_state_maybe_zero_3(model.state_dict(), adapter_args.bias)
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
