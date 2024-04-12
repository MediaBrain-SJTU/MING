import argparse 
import os 
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
from ming.model.builder import load_molora_pretrained_model
from ming.utils import get_model_name_from_path
import torch 
import torch.nn.functional as F
import torch
from ming.conversations import conv_templates, SeparatorStyle
from tqdm import tqdm 
import math
# from ming.utils import client

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def calc_attention_entropy(attention_maps: torch.Tensor, consider_trace=True) -> torch.Tensor:
    # L: 层数, N: 序列长度
    L, bs, heads, N, _ = attention_maps.shape
    attention_maps = attention_maps.squeeze(1).cpu()  # (L, h, N, N)
    # 创建一个下三角矩阵，用于将每层的注意力矩阵转换为下三角形式
    # 这样做是因为attention_maps是下三角的，表示causal attention
    triangular_mask = torch.tril(torch.ones(N, N)).to(attention_maps.device)
    
    # 如果consider_trace为False，那么要把对角线部分也设为mask
    if not consider_trace:
        triangular_mask -= torch.eye(N).to(triangular_mask.device)
    # 应用下三角掩码，去除不需要的上三角部分的注意力值
    attention_maps_triangular = attention_maps * triangular_mask
    
    # 归一化每个token的注意力分布
    # 需要在最后一个维度上求和，并保持维度以便于广播
    if not consider_trace:
        attention_maps_triangular = attention_maps_triangular[..., 1:, :]
        triangular_mask = triangular_mask[1:]
    sum_attention = attention_maps_triangular.sum(dim=-1, keepdim=True)
    normalized_attention_maps = attention_maps_triangular / sum_attention

    # 计算熵，使用广播避免显式循环
    entropy = -(normalized_attention_maps * torch.log2(normalized_attention_maps + 1e-9)).sum(dim=-1)  # (L, h, N)
    
    # 生成权重
    if consider_trace:
        weights = torch.arange(1, N + 1, dtype=torch.float32).to(attention_maps.device)
    else:
        weights = torch.arange(1, N, dtype=torch.float32).to(attention_maps.device)
    
    # 计算加权平均的熵
    layer_weighted_entropy = (entropy * weights).sum(dim=-1) / weights.sum()  # (L, h)
    
    return layer_weighted_entropy.mean(-1), entropy.mean(1)  # (L,), (L, N)

def calc_attention_deviation(attention_maps: torch.Tensor, consider_trace=True) -> torch.Tensor:
    # L: 层数, N: 序列长度
    attention_maps = attention_maps.to(torch.float32)
    L, bs, heads, N, _ = attention_maps.shape
    attention_maps = attention_maps.squeeze(1)  # (L, h, N, N)
    # 创建一个下三角矩阵，用于将每层的注意力矩阵转换为下三角形式
    # 这样做是因为attention_maps是下三角的，表示causal attention
    triangular_mask = torch.tril(torch.ones(N, N)).to(attention_maps.device)
    
    # 如果consider_trace为False，那么要把对角线部分也设为mask
    if not consider_trace:
        triangular_mask -= torch.eye(N).to(triangular_mask.device)
    # 应用下三角掩码，去除不需要的上三角部分的注意力值
    attention_maps_triangular = attention_maps * triangular_mask
    
    # 归一化每个token的注意力分布
    # 需要在最后一个维度上求和，并保持维度以便于广播
    if not consider_trace:
        attention_maps_triangular = attention_maps_triangular[..., 1:, :]
        triangular_mask = triangular_mask[1:]
    sum_attention = attention_maps_triangular.sum(dim=-1, keepdim=True)
    normalized_attention_maps = attention_maps_triangular / (sum_attention + 1e-9)


    mean_attention = normalized_attention_maps.sum(dim=-1, keepdim=True) / (triangular_mask.sum(dim=-1, keepdim=True) + 1e-9)  # (L, h, N - 1 or N, 1)
    std_deviation = torch.sqrt(((normalized_attention_maps - mean_attention) ** 2 * triangular_mask).sum(dim=-1) / (triangular_mask.sum(dim=-1) + 1e-9))  # (L, h, N)

    # 生成权重
    if consider_trace:
        weights = torch.arange(1, N + 1, dtype=torch.float32).to(attention_maps.device)
    else:
        weights = torch.arange(1, N, dtype=torch.float32).to(attention_maps.device)
    
    # 计算加权平均的熵
    layer_weighted_entropy = (std_deviation * weights).sum(dim=-1) / weights.sum()  # (L, h)
    
    return layer_weighted_entropy  # (L, h)

def main(model_path: str, model_base: str, input_file: str, args: argparse.Namespace):
    model_name = get_model_name_from_path(model_path)
    
    if "womolora" in input_file:
        only_load = "attn"
    else:
        only_load = None

    tokenizer, model, context_len, _ = load_molora_pretrained_model(model_path, model_base, 
                                                                    model_name=model_name, only_load=only_load)
    model.eval()
    model.config.output_attentions = True
    # load the input jsonl file
    contents = [json.loads(x) for x in open(input_file, encoding='utf-8', mode='r')]
    contents = get_chunk(contents, args.num_chunks, args.chunk_idx)

    if not args.resume:
        output = open(args.output_file, "w")
        if args.chunk_idx == 0:
            if not args.only_last_layer:
                output.write("\t".join([f"layer_{i + 1}" for i in range(model.config.num_hidden_layers)]) + "\n")
            else:
                output.write(f"layer_{model.config.num_hidden_layers}\n")
    else:
        if args.resume:
            output = open(args.output_file, "a")
            # check the number of lines in output_file, and subtract corresponding number of questions in contents
            with open(args.output_file, 'r') as f:
                num_lines = len(f.readlines())
            num_lines -= 1
            contents = contents[num_lines:]
    
    metric = calc_attention_entropy if args.metric == 'entropy' else calc_attention_deviation
    # import pdb
    # pdb.set_trace()
    for i, content in tqdm(enumerate(contents), total=len(contents), desc='Computing attention:'):

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], content['prompt'])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = tokenizer(content["text"], return_tensors="pt").input_ids
        input_length = len(input_ids[0])
        input_ids = torch.cat([input_ids, output_ids], -1).to(model.device, non_blocking=True)

        with torch.no_grad():
            outputs = model(input_ids)
            attentions = outputs.attentions
        # compute layer weighted fluctuation of attention
        if args.only_last_layer:
            attentions = attentions[-1].unsqueeze(-1)
        else:
            attentions = torch.stack(attentions, dim=0)

        # import pdb
        # pdb.set_trace()
        layer_weighted_entropy, layer_entropy = metric(attentions, True) # (L), (L, N)
        layer_average_entropy = layer_entropy.mean(-1)  # (L)
        layer_max_entropy = layer_entropy.max(-1)[0]  # (L)
        input_layer_average_entropy = layer_entropy[:, :input_length].mean(-1) # (L, )
        input_layer_max_entropy = layer_entropy[:, :input_length].max(-1)[0] # (L, )
        output_layer_average_entropy = layer_entropy[:, input_length:].mean(-1) # (L, )
        output_layer_max_entropy = layer_entropy[:, input_length:].max(-1)[0]# (L, )
        first_output_layer_entropy = layer_entropy[:, input_length] # (L, )

        # save to a text file, where each layer's figure is separated by a tab and each figure maintains two decimal places
        # each sample consistute a line
        output_contents = "\t".join([f"{layer_weighted_entropy[j]:.4f},{layer_average_entropy[j]:.4f},{layer_max_entropy[j]:.4f},{input_layer_average_entropy[j]:.4f},{input_layer_max_entropy[j]:.4f},{output_layer_average_entropy[j]:.4f},{output_layer_max_entropy[j]:.4f},{first_output_layer_entropy[j]:.4f}" for j in range(layer_weighted_entropy.size(0))])
        output.write(output_contents + "\n")
        output.flush()
    
    # close the output file
    output.close()

def test(model_path: str, model_base: str, input_file: str, args: argparse.Namespace):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, context_len, _ = load_molora_pretrained_model(model_path, model_base, 
                                                                    model_name=model_name)
    model.eval()
    model.config.output_attentions = True
    with torch.no_grad():
        input_str = "I am a student."
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(model.device)
        outputs = model(input_ids)
        attentions = outputs.attentions
        attentions = torch.stack(attentions, dim=0)
        print(attentions.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv_mode", type=str, default="qwen")
    parser.add_argument("--only_last_layer", action="store_true")
    parser.add_argument("--metric", type=str, choices=['entropy', 'deviation'], default='entropy')
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_file = args.output_file[:-4] + f"_{args.metric}.csv"
    main(args.model_path, args.model_base, args.input_file, args)
    # with open(args.input_file, "r") as f:
    #     data = json.load(f)
    
    # with open(args.output_file, "w") as f:
    #     for i, d in enumerate(data):
    #         f.write(f"{i}\n")
    #         f.write(f"Input: {d['input']}\n")
    #         f.write(f"Output: {d['output']}\n")
    #         f.write(f"Attention: {d['attention']}\n")
    #         f.write("\n")
    
    # print(f"Done! Output written to {args.output_file}")