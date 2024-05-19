import argparse
import torch
import os
import json
from tqdm import tqdm, trange
# import shortuuid

from ming.conversations import conv_templates, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model
from ming.utils import disable_torch_init, get_model_name_from_path
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import openai
from openai import OpenAI

# from PIL import Image
import math

client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
# print(f"Use OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}!")

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
        # print(line)
        question = line['conversations'][0]['value']
        answer = line['conversations'][1]['value'] if len(line['conversations']) > 1 else None

        additional_info = line['eval']

        
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

def convert_to_json(questions):
    questions = questions.to_dict(orient='records')
    return questions

def eval_model(args):
    if args.question_file.endswith(".csv"):
        questions = pd.read_csv(args.question_file)
        questions = convert_to_json(questions)
    elif args.question_file.endswith(".jsonl"):
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    else:
        # a json file
        with open(args.question_file, 'r') as f:
            questions = json.load(f)
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    if args.resume and os.path.exists(answers_file):
        current_file_num = 0
        with open(answers_file, 'r') as f:
            for line in f:
                current_file_num += 1
        questions = questions[current_file_num:]
        ans_file = open(answers_file, "a", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')
    
    # test
    # questions = questions[:5]

    def get_logit_bias(state_num=4):
        return {(32+i):100 for i in range(state_num)}

    sequence_bias = {}
    task_specific_prompt = ""
    if args.question_file.split("/")[-1].split(".")[0] == 'apps':
        task_specific_prompt = "\n\nPlease use python language to answer this problem. You should process stdin and stdout with input() and print():"
    elif args.question_file.split("/")[-1].split(".")[0] == 'bbh':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}. Let's think step by step."
    elif args.question_file.split("/")[-1].split(".")[0] == 'gsm8k':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."
    elif args.question_file.split("/")[-1].split(".")[0] == 'mmedbench_en_cot':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."
    elif args.question_file.split("/")[-1].split(".")[0] in ['mmedbench_zh_cot', "PLE_Pharmacy_cot", "PLE_TCM_cot"]:
        task_specific_prompt = "\n\n请在回答的最后用以下格式回答：答案为{answer}。"
    elif args.question_file.split("/")[-1].split(".")[0] == 'math' or args.question_file.split("/")[-1].split(".")[0] == 'math_500':
        # task_specific_prompt = "\n\nPlease wrap your final answer with \\box{}."
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as:  The answer is {answer}."
    elif args.question_file.split("/")[-1].split(".")[0] in ["winogrande"]:
        sequence_bias = get_logit_bias(2)
        args.max_new_tokens = 1
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    elif args.question_file.split("/")[-1].split(".")[0] in ["race_high", "race_middle", "mmedbench_en", "mmlu", "arc"]:
        sequence_bias = get_logit_bias(4)
        args.max_new_tokens = 1
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    elif args.question_file.split("/")[-1].split(".")[0] in ["mmedbench_zh", "ceval", "cmmlu", "PLE_Pharmacy", "PLE_TCM"]:
        sequence_bias = get_logit_bias(4)
        args.max_new_tokens = 1
        task_specific_prompt = "\n\n请用选项的字母直接回答，不要输出其他信息："
    elif args.question_file.split("/")[-1].split(".")[0] == "humaneval":
        task_specific_prompt = "\n\nPlease complete the code within the code block ```python```."
        # task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    dataset = CustomDataset(questions)

    for idx in trange(len(dataset)):
        line = dataset[idx]
        question, answer, additional_info = line
        question = question + task_specific_prompt

        message = [{"role": "user", "content": question}]
        completion = client.chat.completions.create(
                    model=args.model,
                    messages=message,
                    temperature=args.temperature,
                    seed=0,
                    logit_bias=sequence_bias,
                    max_tokens=args.max_new_tokens,
                ) 
        outputs = completion.choices[0].message.content

        # ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"prompt": question,
                                   "text": outputs,
                                   "solution": answer,
                                   "additional_info": additional_info,
                                   "model_id": args.model,
                                   "metadata": {}}, ensure_ascii=False) + "\n",)
        ans_file.flush()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    eval_model(args)