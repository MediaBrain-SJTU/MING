import argparse
import torch
import os
import json
from tqdm import tqdm, trange
# import shortuuid
from ming.utils import client

from ming.conversations import conv_templates, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model
from ming.utils import disable_torch_init, get_model_name_from_path
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

# from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset:
    def __init__(self, questions):
        self.questions = questions
        # self.tokenizer = tokenizer
        # self.model_config = model_config
        self.index = 0

    def __getitem__(self, index):
        line = self.questions[index]
        
        # return question, ansewr, additional info
        question = line['conversations'][0]['value']
        answer = line['conversations'][1]['value'] if len(line['conversations']) > 1 else None

        additional_info = line['eval']
        additional_info['id'] = line['id']
        
        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return question, answer, additional_info

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        # 返回迭代器对象本身
        return self
    
    def __next__(self):
        if self.index < len(self.questions):
            # 返回下一个值并更新索引
            item = self.questions[self.index]
            self.index += 1
            return item
        else:
            # 没有更多元素时抛出StopIteration异常
            raise StopIteration


# DataLoader
def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def convert_to_json(questions):
    # questions is a pandas dataframe, which is to be converted to a list object
    # each element in the list is a dictionary
    # the column name of questions is the key of the dictionary
    # the value of the dictionary is the value of the corresponding column
    questions = questions.to_dict(orient='records')
    return questions


def format_self_distillation(instruction: str, original_response: str, *, template: str):
    output = template.format(instruction=instruction,
                             original_response=original_response)
    return output

def eval_model(args):
    # Model
    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # else:
    if "molora" in model_path:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_molora_pretrained_model(model_path, args.model_base, model_name, args.load_molora, use_logit_bias=args.use_logit_bias)
    else:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_pretrained_model(model_path, args.model_base, model_name, use_logit_bias=args.use_logit_bias)

    # load args.question_file, which is a csv file
    if args.question_file.endswith(".csv"):
        questions = pd.read_csv(os.path.expanduser(args.question_file))
        questions = convert_to_json(questions) 
    elif args.question_file.endswith(".jsonl"):
        questions = client.read_jsonl(os.path.expanduser(args.question_file))
    else:
        # a json file
        questions = client.read_json(os.path.expanduser(args.question_file))

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    

    answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    if args.resume:
        current_file_num = 0
        if client.exists(answers_file):
        # if os.path.exists(answers_file):
            data = client.read_jsonl(answers_file)
            current_file_num = len(data)
            # with open(answers_file, 'r') as f:
            #     for line in f:
            #         current_file_num += 1
            questions = questions[current_file_num:]
            ans_file = open(answers_file, "a", encoding='utf-8')
        else:
            ans_file = open(answers_file, "w", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # data_loader = create_data_loader(questions, tokenizer, model.config)
    model: torch.nn.Module
    sequence_bias = None
    def get_tokens_as_tuple(word):
        return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])
    # for name, layer in model.named_modules():
    #     layer.__name__ = name
    #     if "gate_proj" in name:
    #         layer.register_forward_hook(
    #             lambda layer, input, output: print(f"{layer.__name__}: {input[0].shape} {output.shape}")
    #         )
    #         # print(f"register {layer.__name__} hook")
    #         break
    max_new_tokens = 2048

    dataset = CustomDataset(questions)
    with open(args.template_path, "r", encoding='utf-8') as f:    
        template_words = f.read()
    
    for idx in trange(len(dataset), desc="Distilling the dataset"):
    # for idx in trange(10):
        line = dataset[idx]


        question, answer, additional_info = line
        # question = question + task_specific_prompt
        cur_prompt = format_self_distillation(question, answer, template=template_words)
        

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                sequence_bias=sequence_bias)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # ans_id = shortuuid.uuid()
        id_ = additional_info['id']
        del additional_info['id']
        output_dict = {
            "id": id_,
            'conversations': [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": outputs
                }
            ],
            "eval": additional_info
        }

        ans_file.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
        ans_file.flush()
        
    ans_file.close()
    with open(os.path.expanduser(args.answers_file), 'r') as f:
        client.write_jsonl([x for x in f], args.s3_answers_file)
    if not args.keep_local:
        os.remove(os.path.expanduser(args.answers_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-molora", action="store_true")
    parser.add_argument("--template-path", type=str, default="templates/")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--s3-answers-file", type=str, default="s3://syj_test/answer.jsonl")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-logit-bias", action='store_true')
    parser.add_argument("--logit-score", default=15.0)
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)