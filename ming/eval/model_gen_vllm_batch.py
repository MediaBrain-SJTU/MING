import argparse
import torch
import os
import json
from tqdm import tqdm, trange
# import shortuuid


from ming.conversations import conv_templates, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model
from ming.utils import disable_torch_init, get_model_name_from_path
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from vllm import LLM, SamplingParams
# from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, logit_bias):
        self.logit_bias = logit_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for index in self.logit_bias.keys():
            scores[index] += self.logit_bias[index]
        return scores


# Custom dataset class
class CustomDataset:
    def __init__(self, questions, batch_size, conv_mode, task_specific_prompt):
        self.questions = questions
        self.batch_size = batch_size
        self.size = len(questions)
        self.conv = conv_templates[conv_mode].copy()
        self.task_specific_prompt = task_specific_prompt

    def __getitem__(self, index):
        bz = self.batch_size

        # return question, ansewr, additional info
        questions = []
        prompts = []
        answers = []
        additional_infos = []
        for i in range(index*bz, (index+1)*bz):
            if i < self.size:
                conv = self.conv.copy()

                line = self.questions[i]
                question = line['conversations'][0]['value']
                question = question if question.endswith(self.task_specific_prompt) else question + self.task_specific_prompt

                questions.append(question)
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())
                answers.append(line['conversations'][1]['value'] if len(line['conversations']) > 1 else None)
                additional_infos.append(line['eval'] if 'eval' in line else None)

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return questions, prompts, answers, additional_infos

    def __len__(self):
        return len(self.questions) // self.batch_size + 1

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
    questions = questions.to_dict(orient='records')
    return questions

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = LLM(model=args.model_path, trust_remote_code=True)

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

    sequence_bias = None
    def get_tokens_index(word):
        return tokenizer([word], add_special_tokens=False).input_ids[0][0]

    task_specific_prompt = ""
    dataset_name = args.question_file.split("/")[-1].split(".")[0]
    if dataset_name == 'mmedbench_en_cot':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."
    elif dataset_name in ['mmedbench_zh_cot', "PLE_Pharmacy_cot", "PLE_TCM_cot"]:
        task_specific_prompt = "\n\n请在回答的最后用以下格式回答：答案为{answer}。"
    elif dataset_name == 'math' or dataset_name == 'math_500':
        # task_specific_prompt = "\n\nPlease wrap your final answer with \\box{}."
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as:  The answer is {answer}."
    elif dataset_name in ["winogrande"]:
        sequence_bias = {get_tokens_index(x): args.logit_score for x in ["A", "B"]}
        args.max_new_tokens = 1
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    elif dataset_name in ["race_high", "race_middle", "mmedbench_en", "mmlu", "arc"]:
        sequence_bias = {get_tokens_index(x): args.logit_score for x in ["A", "B", "C", "D"]}
        args.max_new_tokens = 1
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    elif dataset_name in ["mmedbench_zh", "ceval", "cmmlu", "PLE_Pharmacy", "PLE_TCM"]:
        sequence_bias = {get_tokens_index(x): args.logit_score for x in ["A", "B", "C", "D"]}
        args.max_new_tokens = 1
        task_specific_prompt = "\n\n请用选项的字母直接回答，不要输出其他信息："
    elif dataset_name == "humaneval":
        task_specific_prompt = "\n\nPlease complete the code within the code block ```python```."
        # task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    dataset = CustomDataset(questions, batch_size=args.batch_size, conv_mode=args.conv_mode, task_specific_prompt=task_specific_prompt)
    for idx in trange(len(dataset)):
        questions, prompts, answers, additional_infos = dataset[idx]
        if len(questions) == 0:
            break
        
        # stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        # stop_str = conv_templates[args.conv_mode].sep2 if conv_templates[args.conv_mode].sep_style == SeparatorStyle.TWO else conv_templates[args.conv_mode].sep
        if args.conv_mode == 'llama3':
                stop_str = [
                    tokenizer.eos_token,
                    "<|eot_id|>"
                ]
        else:
            stop_str = [tokenizer.eos_token]

        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens, stop=stop_str)
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]
        # outputs = outputs[0].outputs[0].text.strip()

        if "cot" in dataset_name:
            if "The answer is" in prompts[0]:
                answer_prompt = "\nThe answer is "
            elif "答案为" in prompts[0]:
                answer_prompt = "\n答案为"
            
            cot_prompts = [(prompt + output + f"{' ' if output.strip().endswith('.') else '. '}{answer_prompt}") for prompt, output in zip(prompts, outputs)]
            # input_token_len = input_ids.shape[1]
            
            if dataset_name not in ["CMExam_cot", "PLE_TCM_cot", "PLE_Pharmacy_cot"]:
                if "E." in prompts[0] or "(E)" in prompts[0]:
                    cot_sequence_bias = {get_tokens_index(x): args.logit_score for x in ["A", "B", "C", "D", "E"]}
                else:
                    cot_sequence_bias = {get_tokens_index(x): args.logit_score for x in ["A", "B", "C", "D"]}
                cot_max_new_tokens = 1
            else:
                cot_sequence_bias = None
                cot_max_new_tokens = 10
            
            if cot_sequence_bias is not None:
                logits_processor_list = LogitsProcessorList([
                    LogitBiasLogitsProcessor(cot_sequence_bias),
                ])
            else:
                logits_processor_list = None

            sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=cot_max_new_tokens, stop=stop_str, logits_processors=logits_processor_list)
            answer_outputs = model.generate(prompts=cot_prompts, sampling_params=sampling_params)
            answer_outputs = [output.outputs[0].text.strip() for output in answer_outputs]
            outputs = [f"{output}{' ' if output.strip().endswith('.') else '. '}{answer_prompt}{answer_output}" for output, answer_output in zip(outputs, answer_outputs)]

        for question, output, answer, additional_info in zip(questions, outputs, answers, additional_infos):
            ans_file.write(json.dumps({"prompt": question,
                                    "text": output,
                                    "solution": answer,
                                    "additional_info": additional_info,
                                    "model_id": model_name,
                                    "metadata": {}}, ensure_ascii=False) + "\n",)
        ans_file.flush()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--logit-score", default=100.0)
    parser.add_argument("--use_logit_bias", action="store_true", default=True)
    parser.add_argument("--only_load", choices=["attn", "ffn"], default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--expert_selection", choices=["topk", "sampling"], default=None)
    args = parser.parse_args()

    eval_model(args)