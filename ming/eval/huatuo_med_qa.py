import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid


from ming.conversations import conv_templates, SeparatorStyle
from ming.model.builder import load_pretrained_model
from ming.utils import disable_torch_init, get_model_name_from_path
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from transformers.generation.utils import GenerationConfig
# from PIL import Image
import math
from multiprocessing import Pool, cpu_count, Manager
from multiprocessing import get_context
import numpy as np 
from functools import partial

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, model_config):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        question = line['Question']
        options = line['Options']
        qs = question + "\n" + options + "\n"
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def convert_to_json(questions):
    questions = questions.to_dict(orient='records')
    return questions

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # load args.question_file, which is a csv file
    questions = pd.read_csv(args.question_file)
    questions = convert_to_json(questions)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if args.resume:
        current_file_num = 0
        if os.path.exists(answers_file):
            with open(answers_file, 'r') as f:
                for line in f:
                    current_file_num += 1
            questions = questions[current_file_num:]
            # ans_file.close()
            ans_file = open(answers_file, "a", encoding='utf-8')
        else:
            ans_file = open(answers_file, "w", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')
        
    # data_loader = create_data_loader(questions, tokenizer, model.config)
    generation_config = GenerationConfig.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B")
    generation_config.max_new_tokens = 128
    generation_config.temperature = args.temperature if args.temperature > 0 else 0.2
    generation_config.do_sample = False if args.temperature == 0 else True
    generation_config.num_beams = args.num_beams
    generation_config.top_p = args.top_p
    model.generation_config = generation_config
    for line in tqdm(questions, total=len(questions)):
        cur_prompt = line['Question'] + "\n" + line['Options'] + "\n"
        cur_prompt += "请回答上述问题：\n"

        messages = []
        messages.append({"role": "user", "content": cur_prompt})
        with torch.inference_mode():
            response = model.HuatuoChat(tokenizer, messages)

        outputs = response.strip()
        ans_file.write(json.dumps({"prompt": cur_prompt,
                                   "text": outputs,
                                   "answer": line['Answer'],
                                   "explanation": line['Explanation'],
                                   'difficulty_level': line['Difficulty level'],
                                   "model_id": model_name,
                                   "metadata": {}}, ensure_ascii=False) + "\n",)
        # ans_file.flush()
    ans_file.close()



def merge_temp_files(args, num_processes):
    answers_file = os.path.expanduser(args.answers_file)
    with open(answers_file, "a") as final_file:
        for i in range(num_processes):
            temp_file_path = answers_file + f".temp{i}"
            with open(temp_file_path, "r") as temp_file:
                for line in temp_file:
                    final_file.write(line)
            os.remove(temp_file_path)  # Delete the temp file after merging
            
def process_questions(model, tokenizer, line,):

    cur_prompt = line['Question'] + "\n" + line['Options'] + "\n"
    messages = [{"role": "user", "content": cur_prompt}]
    with torch.inference_mode():
        response = model.HuatuoChat(tokenizer, messages)
    outputs = response.strip()
    result = json.dumps({"prompt": cur_prompt,
                            "text": outputs,
                            "answer": line['Answer'],
                            "explanation": line['Explanation'],
                            'difficulty_level': line['Difficulty level'],
                            "model_id": 'HuatuoGPT2-7b',
                            "metadata": {}})

    # Write to a temporary file unique to this process
    # temp_answers_file = os.path.expanduser(args.answers_file + f".temp{process_id}")
    # with open(temp_answers_file, "w") as f:
    #     for answer in answers:
    #         f.write(answer + "\n")
    return result
            
def eval_model_parallel(args):
    questions = pd.read_csv(args.question_file)
    questions = convert_to_json(questions)
    answers_file = os.path.expanduser(args.answers_file)
    if args.resume:
        current_file_num = 0
        with open(answers_file, 'r') as f:
            for line in f:
                current_file_num += 1
        questions = questions[current_file_num:]
    num_processes = 4  # Or a fixed number of processes if preferred
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.generation_config = GenerationConfig.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B")




    # Divide the questions into chunks for each process
    # questions_chunks = np.array_split(questions, num_processes)
    # pbar = tqdm(total=len(questions))
    # pbar.set_description('HuatuoGPT2-Inference:')
    # update = lambda *args: pbar.update()
    # Use multiprocessing to process questions in parallel
    ctx = get_context('spawn')  # 使用'spawn'方式启动进程
    pool = ctx.Pool(num_processes)
    result = []
    partial_func = partial(process_questions, model, tokenizer)
    result = list(tqdm(pool.imap(partial_func, questions), total=len(questions)))
    # result = pool.map_async(partial_func, questions, callback=lambda x: pbar.close())
    pool.close()
    pool.join()
    # for q in questions:
        # result.append(pool.apply_async(process_questions, (model, tokenizer, q, args, 0), callback=update))
    result = result.get()
    for each in result:
        with open(answers_file, "w", encoding='utf-8') as f:
            f.write(each + "\n")
    # with ctx.Pool(num_processes) as p:
    #     p.starmap(process_questions, [(model, tokenizer, questions_chunks[i], args, i) for i in range(num_processes)])
    # with Manager() as manager:
    #     progress_bar = manager.list([0])  # Create a managed list to track progress
    #     total = len(questions)  # Total number of questions to process
        
    #     with Pool(num_processes) as p, tqdm(total=total) as pbar:
    #         # Wrap the call to starmap with a list comprehension to ensure the pool waits for all tasks
    #         [p.apply_async(process_questions, (questions_chunks[i], args, i, pbar)) for i in range(num_processes)]
    #         p.close()  # Close the pool
    #         p.join()  # Wait for all processes to finish
    # Merge temporary files into the final answers file
    merge_temp_files(args, num_processes)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
