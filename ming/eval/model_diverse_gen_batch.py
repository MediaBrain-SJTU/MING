import argparse
import torch
import os
import json
from tqdm import tqdm, trange
# import shortuuid


from ming.conversations import conv_templates, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model, load_pretrained_orth_model
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

def get_loss(logits, labels, attention_mask, vocab_size):
    from torch.nn import CrossEntropyLoss
    labels = labels.masked_fill(~attention_mask, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    B, N, C = shift_logits.shape
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # this loss is [-1, ], we need to reshape it to [B, N]
    loss = loss.reshape(B, N)
    # we must know that some positions are 0-loss because of ignore_index, we need to ignore these
    loss_sum = loss.sum(dim=-1)
    loss_actual_position = torch.not_equal(loss, 0).sum(dim=-1)
    loss = loss_sum / loss_actual_position  # [B, ]
    return loss


def generate_func(model, input_ids, **kwargs):
    if input_ids.dim() == 1:
        # only one item
        input_ids = input_ids.unsqueeze(0)
    max_new_tokens = kwargs.pop("max_new_tokens", args.max_new_tokens)
    tokenizer = kwargs.pop("tokenizer")
    sequence_bias = kwargs.pop("sequence_bias", None)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer("<|im_end|>")["input_ids"][-1],
            sequence_bias=sequence_bias,
            use_cache=True)
    return output_ids

def switch_expert(one_model, other_model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  **kwargs):
    # calculate the loss of input_ids on base_model
    print(input_ids.shape)
    with torch.inference_mode():
        one_model_logits = one_model(input_ids).logits
        one_model_loss = get_loss(one_model_logits, input_ids, attention_mask, one_model.config.vocab_size)
        # calculate the loss of input_ids on loaded_model
        other_model_logits = other_model(input_ids).logits
        other_model_loss = get_loss(other_model_logits, input_ids, attention_mask, other_model.config.vocab_size)

    # based on the loss scale, partition the inputs to different models
    mask = one_model_loss < other_model_loss
    one_model_inputs_id = torch.nonzero(mask).squeeze(-1) # [k, ]
    other_model_inputs_id = torch.nonzero(~mask).squeeze(-1) # [k, ]
    # print(one_model_inputs_id, other_model_inputs_id)
    if one_model_inputs_id.shape[0] == 0:
        one_model_outputs = []
    else:
        one_model_outputs = generate_func(one_model, input_ids[one_model_inputs_id], **kwargs)
    if other_model_inputs_id.shape[0] == 0:
        other_model_outputs = []
    else:
        other_model_outputs = generate_func(other_model, input_ids[other_model_inputs_id], **kwargs)
    if one_model_outputs == []:
        return other_model_outputs 
    if other_model_outputs == []:
        return one_model_outputs
    
    tokenizer = kwargs.pop("tokenizer")
    # concat one_model_outputs and other_model_outputs with tokenizer.eos_token_id 
    max_len = max(one_model_outputs.shape[1], other_model_outputs.shape[1])
    # print(max_len)
    one_model_outputs = torch.cat([one_model_outputs, 
                                   torch.full((one_model_outputs.shape[0], max_len-one_model_outputs.shape[1]), tokenizer.eos_token_id, dtype=torch.long, device=one_model_outputs.device)], dim=1)
    other_model_outputs = torch.cat([other_model_outputs, 
                                     torch.full((other_model_outputs.shape[0], max_len-other_model_outputs.shape[1]), tokenizer.eos_token_id, dtype=torch.long, device=other_model_outputs.device)], dim=1)
    # print(one_model_outputs.shape, other_model_outputs.shape)
    total_outputs = torch.cat([one_model_outputs, other_model_outputs], dim=0)
    # merge the outputs
    results = torch.zeros(total_outputs.shape[0], total_outputs.shape[1]).to(total_outputs)
    
    results = torch.scatter_add(results, 0, torch.cat([one_model_inputs_id, other_model_inputs_id], dim=0).unsqueeze(1).expand(-1, total_outputs.shape[1]), total_outputs)
    # except Exception as e:
    #     print(e)
    #     print(one_model_inputs_id)
    #     print(other_model_inputs_id)
    #     exit(-1)
    # if results.shape[0] != input_ids.shape[0]:
    
    return results
    

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
                questions.append(question)
                conv.append_message(conv.roles[0], question+self.task_specific_prompt)
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
    # questions is a pandas dataframe, which is to be converted to a list object
    # each element in the list is a dictionary
    # the column name of questions is the key of the dictionary
    # the value of the dictionary is the value of the corresponding column
    questions = questions.to_dict(orient='records')
    return questions

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.question_file.split("/")[-1].split(".")[0] in ["mmedbench_zh", "ceval", "cmmlu", "race_high", "race_middle", "mmedbench_en", "mmlu", "arc", "winogrande"]:
        args.use_logit_bias = True
    
    # import pdb
    # pdb.set_trace()

    # else:
    if "orth" in model_path:
        tokenizer, model, context_len, tokenizer_with_prefix_space, other_model = load_pretrained_orth_model(model_path, args.model_base, args.lora_name_or_path, use_logit_bias=args.use_logit_bias, only_load=args.only_load, switch_experts=args.switch_old_expert)
    elif "molora" in model_path:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_molora_pretrained_model(model_path, args.model_base, model_name, use_logit_bias=args.use_logit_bias, only_load=args.only_load, expert_selection=args.expert_selection)
    else:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_pretrained_model(model_path, args.model_base, model_name, use_logit_bias=args.use_logit_bias, only_load=args.only_load)
    tokenizer.padding_side = "left"
    tokenizer_with_prefix_space.padding_side = "left"

    # load args.question_file, which is a csv file
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
    # import pdb
    # pdb.set_trace()

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

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # data_loader = create_data_loader(questions, tokenizer, model.config)
    model: torch.nn.Module
    model.eval()
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
    task_specific_prompt = ""
    dataset_name = args.question_file.split("/")[-1].split(".")[0]
    if dataset_name == 'apps':
        task_specific_prompt = "\n\nPlease use python language to answer this problem. You should process stdin and stdout with input() and print():"
    elif dataset_name == 'bbh':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}. Let's think step by step."
    elif dataset_name == 'gsm8k':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."
    elif dataset_name == 'mmedbench_en_cot':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."
    elif dataset_name in ['mmedbench_zh_cot', "PLE_Pharmacy_cot", "PLE_TCM_cot"]:
        task_specific_prompt = "\n\n请在回答的最后用以下格式回答：答案为{answer}。"
    elif dataset_name == 'math' or dataset_name == 'math_500':
        # task_specific_prompt = "\n\nPlease wrap your final answer with \\box{}."
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as:  The answer is {answer}."
    elif dataset_name in ["winogrande"]:
        sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B"]}
        args.max_new_tokens = 1
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    elif dataset_name in ["race_high", "race_middle", "mmedbench_en", "mmlu", "arc"]:
        sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
        args.max_new_tokens = 1
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    elif dataset_name in ["mmedbench_zh", "ceval", "cmmlu", "PLE_Pharmacy", "PLE_TCM"]:
        sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
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

        # print("[FIRST INPUT]: ", prompt)
        input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
        if args.switch_old_expert:
            assert other_model is not None, "please specify the --switch_old_expert argument"
            output_ids = switch_expert(model, other_model, input_ids, attention_mask,
                                       tokenizer=tokenizer, 
                                       sequence_bias=sequence_bias,
                                       max_new_tokens=args.max_new_tokens,)
        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer("<|im_end|>")["input_ids"][-1],
                    sequence_bias=sequence_bias,
                    use_cache=True)
        # print(input_ids.shape, output_ids.shape)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # print("original outputs: ",tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True) )
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # print("cut outputs: ", outputs)
        
        # print("[FIRST OUTPUT]: ", outputs)

        if "cot" in dataset_name:
            if "The answer is" in prompts[0]:
                cot_prompt = "\nThe answer is "
            elif "答案为" in prompts[0]:
                cot_prompt = "\n答案为"
            
            conv = conv_templates[args.conv_mode].copy()
            cut_length = len(conv.sep2)
            cot_prompts = [(prompt + output + f"{' ' if output.strip().endswith('.') else '. '}{cot_prompt}") for prompt, output in zip(prompts, outputs)]
            input_ids = tokenizer(cot_prompts, return_tensors='pt', padding=True).input_ids.to(device='cuda', non_blocking=True)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
            
            if dataset_name not in ["CMExam_cot", "PLE_TCM_cot", "PLE_Pharmacy_cot"]:
                if "E." in prompts[0] or "(E)" in prompts[0]:
                    cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D", "E"]}
                else:
                    cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
                cot_max_new_tokens = 1
            else:
                cot_sequence_bias = None
                cot_max_new_tokens = 50
            if args.switch_old_expert:
                assert other_model is not None, "please specify the --switch_old_expert argument"
                
                answer_output_ids = switch_expert(model, other_model, input_ids, attention_mask,
                                                  tokenizer=tokenizer,
                                                  max_new_tokens=cot_max_new_tokens,
                                                  sequence_bias=cot_sequence_bias)
            else:
                with torch.inference_mode():
                    answer_output_ids = model.generate(
                    input_ids,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=cot_max_new_tokens,
                    eos_token_id=tokenizer("<|im_end|>")["input_ids"][-1],
                    sequence_bias=cot_sequence_bias,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != answer_output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            # print("all: ", tokenizer.batch_decode(answer_output_ids, skip_special_tokens=True))
            # print("input: ", cot_prompts)
            # print("all: ", tokenizer.batch_decode(answer_output_ids[:, :], skip_special_tokens=True))
            answer_outputs = tokenizer.batch_decode(answer_output_ids[:, input_token_len:], skip_special_tokens=True)
            # print(answer_outputs)
            outputs = [f"{output}{' ' if output.strip().endswith('.') else '. '}{cot_prompt}{answer_output}" for output, answer_output in zip(outputs, answer_outputs)]


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
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--logit-score", default=100.0)
    parser.add_argument("--use_logit_bias", action="store_true", default=False)
    parser.add_argument("--only_load", choices=["attn", "ffn", "share", "no_share"], default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--expert_selection", choices=["topk", "sampling"], default=None)
    parser.add_argument("--lora_name_or_path", type=str, default=None)
    parser.add_argument("--switch-old-expert", action="store_true", default=None)
    args = parser.parse_args()

    eval_model(args)