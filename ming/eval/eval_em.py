import os 
import json 
import jsonlines

import argparse 
from tqdm import tqdm, trange
from subprocess import PIPE, Popen, TimeoutExpired
import tempfile
import re 
from pathlib import Path
import signal
from sympy import sympify
import pandas as pd
import evaluate
from ming.eval.cblue.post_generate_process import process_generated_results
from ming.eval.cblue.evaluate import calc_scores, report_error_msg, error_msg, report_score

def normalize_frac(x):
    # Pattern to match \frac{a}{b}
    pattern = r'\\frac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize_dfrac(x):
    pattern = r'\\dfrac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize(x):
    if "\\frac" in x and normalize_frac(x):
        a, b = normalize_frac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
        
    elif "\\dfrac" in x and normalize_dfrac(x):
        a, b = normalize_dfrac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
    else:
        try:
            x = sympify(x).evalf()
            return float(x)
        except:
            return x

def acc(pred, target):
    return 1 if pred == target else 0

def rouge(pred, target):
    # compute rouge-1, rouge-2, rouge-l
    pass

def extract_bbox_content(s):
    contents = []
    i = 0
    while i < len(s):
        if s[i:i+7] == '\\boxed{':
            depth = 1
            start = i + 7
            i += 7
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        contents.append(s[start:i])
                i += 1
        else:
            i += 1
    return contents

def extract_answer_content(s):
    match = re.search(r'answer is (.*?)(\.|$)', s, re.IGNORECASE)
    return match.group(1) if match else None

def math_acc(line):
    pred = line['text']
    target = line['additional_info']['solution']

    target_answer = extract_bbox_content(target)[0]
    pred_answer = extract_answer_content(pred)

    if pred_answer is not None:
        return 1 if target_answer in pred_answer else 0
    else:
        return 0

    # print(target)
    # print(target_answer)
    # print(pred)

    # if pred_answer != []:
    #     pred_answer = pred_answer[0]
    #     target_answer = normalize(target_answer)
    #     if isinstance(target_answer, float):
    #         pred_answer = normalize(pred_answer) #if pred_answer is not None else float("-inf")

    #     if isinstance(target_answer, str) and isinstance(pred_answer, str): # target type = str
    #         return 1.0 if target_answer in pred_answer else 0.0
        
    #     elif isinstance(pred_answer, str): # target type = float
    #         return 1.0 if pred_answer in target else 0.0
        
    #     elif isinstance(pred_answer, float):
    #         if abs(pred_answer - target_answer) < 1e-3:
    #             return 1.0
    #         else:
    #             return 0.0
    # else:
    #     if "." in pred:
    #         pred_answer = pred.split(".")[-1]
    #     else:
    #         pred_answer = pred[len(pred) // 2:]

    #     return 1.0 if target_answer in str(pred_answer) else 0.0


def code_acc(line):
    cwd = os.getcwd()
    text = line['text']

    # 示例字符串

    # 使用正则表达式匹配第一对```之间的内容
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    
    # 如果找到匹配项，则提取并打印
    if match:
        extracted_content = match.group(1)
    else:
        extracted_content = text

    additional_info = line['additional_info']
    # function_name = additional_info['function_name']
    test = additional_info['test']
    executable_code = extracted_content
    if isinstance(test, str):
        test_code = executable_code + "\n" + test
    else:
        test_code = executable_code + "\n" + "\n".join(test)
    
    if additional_info.get("entry_point", None) is not None:
        test_code = test_code + "\n\n" + f"check({additional_info['entry_point']})"
    

    with tempfile.TemporaryDirectory() as tempdir_name:
        tempdir_name = Path(tempdir_name)
        with open(tempdir_name / "program.py", "w", encoding="UTF-8") as f:
            f.write(test_code)
        os.chdir(tempdir_name)
        

    # idx = additional_info["id"]
    # with open(f"/remote-home/syjiang/repo/MING-MOE/logs/diverse/humaneval/tmp/{idx}", 'w') as f:
    #     f.write(test_code)
        
        p = Popen(f'python program.py', shell=True, stdout=PIPE, stderr=PIPE)
        time_limit = 15  # seconds
        scores = 1
        try:
            stdout, stderr = p.communicate(timeout=time_limit)
        except TimeoutExpired:
            # Linux
            # os.killpg(p.pid, signal.SIGTERM)
            # Windows
            os.system("kill {pid}".format(pid=p.pid))
            scores = 0
        else:
            if stderr:
                scores = 0
    

    os.chdir(cwd)
    return scores

def gsm8k_acc(line):
    # extract answer after #### 
    pred = line['text']
    target = line['additional_info']['answer']
    index = target.find("####")
    target_answer = target[index + 4:].strip()

    pred_answer = extract_answer_content(pred)
    # import pdb
    # pdb.set_trace()
    # if index != -1:
    #     pred_answer = pred[index + 4:].strip()  # Extract answer after "####" and strip any leading or trailing whitespace
    # else:
    #     pred_answer = pred
    # index = target.find("####")
    # target_answer = target[index + 4:].strip()
    if pred_answer is not None:
        return 1 if target_answer in pred_answer else 0
    else:
        return 0

def answer_acc(line):
    pred = line['text']

    pred = re.findall(r'[A-E]', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer']

    return 1 if pred == answer else 0 

def rte_acc(line):
    pred = line['text']
    mapping = {"A": "Yes", "B": "No"}
    answer = line['additional_info']['answer']

    return 1 if mapping[answer] in pred else 0 

def mmedbench_acc(line):
    pred = line['text']

    pred = re.findall(r'[A-E]', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if pred == answer else 0 

def winogrande_acc(line):
    pred = line['text']
    # import pdb
    # pdb.set_trace()

    option1, option2 = line['prompt'].split("\n")[2:4]
    option1 = option1[3:]
    option2 = option2[3:]

    index1 = pred.find(option1)
    index2 = pred.find(option2)

    if index1 == -1 and index2 != -1:
        pred = "B"
    elif index1 != -1 and index2 == -1:
        pred = "A"
    elif index1 != -1 and index2 != -1:
        if index1 < index2:
            pred = "A"
        else:
            pred = "B"
    else:
        return 1 if line["additional_info"]["answer"] in pred else 0        
    
    answer = line['additional_info']['answer']

    return 1 if pred == answer else 0 



def bbh_acc(line):
    pred = line['text']
    pred = extract_answer_content(pred)
    answer = line['additional_info']['target']
    if "(" in pred and ")" in pred:
        # extract the content in () [maybe many], and select the one which is a single capital letter
        pred = re.findall(r'\((.*?)\)', pred)
        for p in pred:
            if p in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                pred = f"({p})"
                break


    return 1 if answer in pred else 0

def record_acc(line):
    pred = line['text']
    answers = line['additional_info']['answers']
    for answer in answers:
        if answer["text"] in pred:
            return 1
    
    return 0

def apps_acc(line):
    text = line['text']
    match = re.search(r"```python(.*?)```", text)

    # 如果找到匹配项，则提取并打印
    if match:
        extracted_content = match.group(1)
    else:
        extracted_content = text
    additional_info = line['additional_info']
    input_output = additional_info['input_output']
    # try:
    #     input_output = json.loads(input_output)
    # except:
    #     return None
    # input_output = json.loads(input_output)

    inputs = input_output['inputs']
    outputs = input_output['outputs']
    test_code = extracted_content 
    assert len(inputs) == len(outputs)
    
    ff = tempfile.NamedTemporaryFile(mode='w')
    ff.write(test_code)
    name = ff.name 
    scores = 1
    for i in range(len(inputs)):
        cur_input = inputs[i]
        cur_output = outputs[i]
        
        p = Popen(f'python {name} < {cur_input}', shell=True, stdout=PIPE, stderr=PIPE)
        time_limit = 15  # seconds
        try:
            stdout, stderr = p.communicate(timeout=time_limit)
        except TimeoutExpired:
            # Linux
            # os.killpg(p.pid, signal.SIGTERM)
            # Windows
            # Popen("TASKKILL /F /PID {pid} /T".format(pid=p.pid))
            scores = 0
            break
        if stderr:
            scores = 0
            break
        if stdout.strip() != cur_output.strip():
            scores = 0
            break
    ff.close()
    return scores

def triviaqa_acc(line):
    pred = line['text']
    answers = line['additional_info']['answer']
    for answer in answers:
        if pred == answer:
            return 1 
    return 0

def mmedbench_en_cot_acc(line):
    pred = re.search(r'the answer is (.*?)(\.|$)', line['text'], re.IGNORECASE)
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if f"{answer}." in pred else 0 

def mmedbench_zh_cot_acc(line):
    pred = re.search(r'答案为(.*?)$', line['text'])
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if f"{answer}" in pred else 0 

def xsum_rouge(qargs, uestions):
    rouge = evaluate.load('ming/eval/rouge')
    predictions = []
    references = []
    for line in tqdm(questions, total=len(questions)):
        predictions.append(line["text"])
        references.append(line['additional_info']['summary'])
    
    results = rouge.compute(predictions=predictions,
                         references=references)
    
    return results

def squadv20_acc(line):
    for answer in line["additional_info"]["answer"]:
        if answer in line["text"]:
            return 1
    return 0

def cblue_score(args, questions):
    answer_file = "/home/cs/yangyuchen/yushengliao/Medical_LLM/Medical_MOE/datas/medical_moe/test/CBLUE_structured.json"
    output_path = args.output_file.rsplit("/", 1)[0] + "results.json"
    dict_pred = process_generated_results(questions)
    
    dict_gt = json.load(
        open(answer_file, "r", encoding="utf-8")
    )

    score_map, success_flag = calc_scores(
        dict_gt, dict_pred, output_path
    )

    if success_flag:
        # turn to 100-score format
        score_map = {key: value * 100 for key, value in score_map.items()}
        report_score(score_map, output_path)

def multiplechoice_acc(line):
    pred = re.search(r'答案为(.*?)$', line['text'])
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        # import pdb
        # pdb.set_trace()
        answer = line['additional_info']['answer']
        if "错，为本题正确答案" in line['text'] and f"{answer}错，为本题正确答案" in line['text']:
            return 1
        else:
            all_index = "ABCDE"
            for answer in line['additional_info']['answer']:
                all_index = all_index.replace(answer, "")
                if f"{answer}对" not in line['text']:
                    return 0
            # if len(line['additional_info']['answer']) > 1:
            if True:
                for o_answer in all_index:
                    if f"{o_answer}对" in line['text']:
                        return 0
            return 1

    else:
        pred = pred[0]
    
    all_index = "ABCDE"
    answer_list = line['additional_info']['answer']
    for answer in answer_list:
        all_index = all_index.replace(answer, "")
        if answer not in pred:
            return 0
    # if len(answer_list) > 1:
    if True:
        for o_answer in all_index:
            if f"{o_answer}" in pred:
                return 0
    return 1

def multiplechoice_en_acc(line):
    pred = re.search(r'The answer is (.*?)$', line['text'])
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        # import pdb
        # pdb.set_trace()
        return 0
    else:
        pred = pred[0]

    # return 1 if line['additional_info']['answer'] in pred else 0
    
    all_index = "ABCDE"
    answer_list = line['additional_info']['answer']
    for answer in answer_list:
        all_index = all_index.replace(answer, "")
        if f"{answer}." not in pred:
            return 0
        
    if len(answer_list) > 1:
        for o_answer in all_index:
            if f"{o_answer}." in pred:
                return 0
    return 1

METRIC_FUNC_MAPPING = {
    "math": math_acc,
    "math_500": math_acc,
    "humaneval": code_acc,
    "mbpp": code_acc,
    "gsm8k": gsm8k_acc,
    "mmedbench_en": mmedbench_acc,
    "mmedbench_zh": mmedbench_acc,
    "bbh": bbh_acc,
    "apps": apps_acc,
    "triviaqa": triviaqa_acc,
    "race_high": answer_acc,
    "race_middle": answer_acc,
    "mmlu": answer_acc,
    "ceval": answer_acc,
    "cmmlu": answer_acc,
    "rte": rte_acc,
    "winogrande": answer_acc,
    "record": record_acc,
    "arc": answer_acc,
    "mmedbench_en_cot": mmedbench_en_cot_acc,
    "mmedbench_zh_cot": mmedbench_zh_cot_acc,
    "squadv20": squadv20_acc,
    "squadv20_1000": squadv20_acc,
    "PLE_TCM_cot": multiplechoice_acc,
    "PLE_Pharmacy_cot": multiplechoice_acc,
    "PLE_TCM": answer_acc,
    "PLE_Pharmacy": answer_acc,
    "ceval_cot": multiplechoice_acc,
    "cmmlu_cot": multiplechoice_acc,
    "CMExam_cot": multiplechoice_acc,
    "CMB_cot": multiplechoice_acc,
    "mmlu_cot": multiplechoice_en_acc,
    "MedQA_cot": multiplechoice_en_acc,
    "MedMCQA_cot": multiplechoice_en_acc
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)

    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # input_file is a jsonl file with the following format:
    questions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    
    total_num = len(questions)
    total_score = 0
    dataset_name  = args.input_file.split("/")[-2]
    if dataset_name in METRIC_FUNC_MAPPING:
        acc_func = METRIC_FUNC_MAPPING[dataset_name]
    else:
        acc_func = None
    wrong_idx = []

    if dataset_name in ["xsum", "xsum_1000"]:
        # open-end task
        score = xsum_rouge(args, questions)
        print(score)
    elif dataset_name in ["CBLUE"]:
        cblue_score(args, questions)
    else:
        # answer task   
        # import pdb
        # pdb.set_trace()
        if "type" in questions[0]['additional_info']:
            type_total = {}
            type_score = {}
            for line in tqdm(questions, total=total_num):
                class_type = line['additional_info']["type"]
                if class_type not in type_total:
                    type_total[class_type] = 1
                    type_score[class_type] = 0
                else:
                    type_total[class_type] += 1

                scores = acc_func(line)

                if scores is None:
                    type_total[class_type] -= 1
                    wrong_idx.append(line)
                    continue

                type_score[class_type] += scores
                if scores == 0:
                    wrong_idx.append(line)
            
            type_acc = {"type": [], "acc": []}
            for class_type in type_total:
                type_acc["type"].append(class_type)
                type_acc["acc"].append(type_score[class_type] / type_total[class_type])
            
            # import pdb
            # pdb.set_trace()
            avg_acc = sum([type_score[class_type] for class_type in type_total]) / sum([type_total[class_type] for class_type in type_total])
            type_acc["type"].append("AVERAGE")
            type_acc["acc"].append(avg_acc)

            # import pdb
            # pdb.set_trace()
            df = pd.DataFrame(type_acc)
            df.to_csv(os.path.join(args.output_file.rsplit("/", 1)[0], "type_acc.csv"), index=False)
            print(f"Acc in {dataset_name}: {avg_acc}")
            
        else:
            for line in tqdm(questions, total=total_num):
                scores = acc_func(line)
                # print("score: ", scores)
                # import pdb
                # pdb.set_trace()
                if scores is None:
                    total_num -= 1
                    wrong_idx.append(line)
                    continue
                total_score += scores
                if scores == 0:
                    wrong_idx.append(line)
        
            avg_acc = total_score / total_num
            print(f"Acc in {dataset_name}: {avg_acc}")

        with open(args.output_file, 'w') as f:
            json.dump(wrong_idx, f, ensure_ascii=False)