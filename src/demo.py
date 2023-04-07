from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration 
import logging
import os
import math
import json
import torch
from argparse import ArgumentParser

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
        
def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default = "/model")
    parser.add_argument('--peft_path', type=str, default = '/Fine_Tuning_Results/lora.p')
    parser.add_argument('--gpu_id', type=str, default = "0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        
    logger = logging.getLogger(__file__)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_class = ChatGLMForConditionalGeneration 

    logger.info("Setup Model")
    num_layers = read_json(os.path.join(args.model_path , "config.json"))["num_layers"]
    device_ids = list(range(torch.cuda.device_count()))

    device_map = {}
    device_map["transformer.word_embeddings"] = device_ids[0]
    device_map["transformer.final_layernorm"] = device_ids[-1]
    device_map["lm_head"] = device_ids[0]

    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) * math.ceil(num_layers / len(device_ids)))
    ]
    allocations = allocations[len(allocations)-num_layers:]
    for layer_i, device_id in enumerate(allocations):
        device_map[f"transformer.layers.{layer_i}.input_layernorm"] = device_id
        device_map[f"transformer.layers.{layer_i}.attention.rotary_emb"] = device_id
        device_map[f"transformer.layers.{layer_i}.attention.query_key_value"] = device_id
        device_map[f"transformer.layers.{layer_i}.attention.dense"] = device_id
        device_map[f"transformer.layers.{layer_i}.post_attention_layernorm"] = device_id
        device_map[f"transformer.layers.{layer_i}.mlp.dense_h_to_4h"] = device_id
        device_map[f"transformer.layers.{layer_i}.mlp.dense_4h_to_h"] = device_id

    model_class = ChatGLMForConditionalGeneration 
    model = model_class.from_pretrained(args.model_path, device_map = device_map).half()
    model.config.use_cache = True # silence the warnings. Please re-enable for inference!
    logger.info("Setup PEFT")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['query_key_value'],
    )
    model = get_peft_model(model, peft_config)

    for layer_i in range(len(model.base_model.model.transformer.layers)):
        device = model.base_model.model.transformer.layers[layer_i].attention.query_key_value.weight.device
        model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_B.half().to(device)
        model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_A.half().to(device)

    if os.path.exists(args.peft_path ):
            model.load_state_dict(torch.load(args.peft_path), strict=False)

    model.eval()
    print("Human:")
    while True:
        history=[]
        line = input()
        inputs = line.split('[\\n]')
        if len(inputs) > 1:
            for i in range(0,len(inputs)-1,2):
                history = history + [(inputs[i],inputs[i+1])]
        query = inputs[-1]
        if history != []:
            response, history = model.chat(tokenizer, query, history=history)
        else:
            response, history = model.chat(tokenizer, query, history=[])
        print("\n------------------------------------------------\nAnswer:")
        print(response)
        print("\n------------------------------------------------\nHuman:")


if __name__ == '__main__':
    main()