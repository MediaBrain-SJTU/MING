"""Inference for FastChat models."""
import abc
from typing import Optional
import warnings
import time
import torch
from ming.conversations import conv_templates, get_default_conv_template, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model
import numpy as np
import pdb


def compute_skip_echo_len(model_name, conv, prompt):
    model_name = model_name.lower()
    if "chatglm" in model_name:
        skip_echo_len = len(prompt)
    elif "dolly" in model_name:
        special_toks = ["### Instruction:", "### Response:", "### End"]
        prompt_tmp = prompt
        for tok in special_toks:
            prompt_tmp = prompt_tmp.replace(tok, "")
        skip_echo_len = len(prompt_tmp)
    elif "bloom" in model_name:
        skip_echo_len = len(prompt) - prompt.count("</s>") * 4
    else:
        skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
    return skip_echo_len


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device, beam_size,
                    context_len=4096, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.2))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = torch.tensor(input_ids[-max_src_len:]).unsqueeze(0).cuda()

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=beam_size,
        temperature=temperature,
    )
    outputs = outputs[0][len(input_ids[0]):]
    output = tokenizer.decode(outputs, skip_special_tokens=True)

    return output


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


def chat_loop(model_path: str, model_base:str, device: str,
              conv_template: Optional[str], temperature: float,
              max_new_tokens: int, beam_size: int,chatio: ChatIO,
              debug: bool):
    # Model
    # model, tokenizer = load_model(model_path, device,
    #     num_gpus, max_gpu_memory, load_8bit, debug)
    if "moe" in model_path.lower():
        tokenizer, model, context_len, _ = load_molora_pretrained_model(model_path, model_base, None, use_logit_bias=None, only_load=None, expert_selection=None)
    else:
        tokenizer, model, context_len, _ = load_pretrained_model(model_path, model_base, None, use_logit_bias=None, only_load=None)

    model.config.use_cache = True
    model.eval()

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()

    while True:
        while True:
            try:
                # pdb.set_trace()
                inp = chatio.prompt_for_input(conv.roles[0])
                while "new chat" in inp:
                    conv.messages = []
                    inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""
            except UnicodeDecodeError:
                continue
            break

        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        generate_stream_func = generate_stream
        prompt = conv.get_prompt()

        # skip_echo_len = compute_skip_echo_len(model_path, conv, prompt)

        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        context_len = len(prompt)  + max_new_tokens + 8
        T1 = time.time()
        output_stream = generate_stream_func(model, tokenizer, params, device, beam_size,context_len=context_len)
        T2 = time.time()
        if debug:
            print('程序运行时间:%s秒' % ((T2 - T1)))
        # import pdb
        # pdb.set_trace()
        conv.messages[-1][-1] = output_stream
        print(output_stream)
        # outputs = chatio.stream_output(output_stream, skip_echo_len)
        # # NOTE: strip is important to align with the training data.
        # conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "output_stream": output_stream, "len":len(prompt)}, "\n")
