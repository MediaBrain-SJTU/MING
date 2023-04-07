import os
from multiprocessing import Pool, Lock
import torch
import numpy as np
from random import choices
import time
import pandas as pd
import torch
from tqdm import tqdm
import os
import openai
import json
from progress.bar import Bar
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from rouge_score import rouge_scorer

def parse_response(respones):
    import re
    original=respones
    respones = re.split("###", respones)[0]
    intruction_pattern = re.compile(r"(?<=(?:" + '|'.join(['指令:', '指令：']) + "))[\s\S]*?(?=" + '|'.join(['输入:', '输入：']) + ")")
    input_pattern = re.compile(r"(?<=(?:" + '|'.join(['输入:', '输入：']) + "))[\s\S]*?(?=" + '|'.join(['输出:', '输出：']) + ")")
    output_pattern = re.compile(r"(?<=(?:" + '|'.join(['输出:', '输出：']) + "))[\s\S]*?(?=$)")
    intruction_match = intruction_pattern.search(respones)
    input_match = input_pattern.search(respones)
    output_match = output_pattern.search(respones)
    if intruction_match and input_match and output_match:
        inst = re.sub(r'\d+\.$', '', intruction_match.group().strip()).strip('\n').rstrip()
        input = re.sub(r'\d+\.$', '', input_match.group().strip()).strip('\n').rstrip()
        input = "" if "无输入" in input else input
        output = output_match.group().strip().strip('\n')
        if '指令:' in output and '输入:' in output and '输出:' in output: # 返回若没有以###号区分，取第一条数据
            output_pattern_new = re.compile(r"(?<=(?:" + "))[\s\S]*?(?=" + '|'.join(['指令:', '指令：']) + ")")
            output_match_new = output_pattern_new.search(output)
            if output_match_new:
                output = re.sub(r'\d+\.$', '', output_match_new.group().strip()).strip('\n').rstrip()
        return {"original":original,"instruction":inst, "input":input, "output":output}
    else:
        return None

def gpt_generate(i):
   
    global count
  
    lock.acquire()
    count=count+1
    openai.api_key = api_keys[count%len(api_keys)]
    lock.release()

    task_subset=choices(seed_task_list,k=6)
    prompt_=prompt
    for idx, task_dict in enumerate(task_subset):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        input = "<无输入>" if input.lower() == "" else input
        prompt_ =prompt_+ f"###\n"
        prompt_ += f"{idx + 1}. 指令: {instruction}\n"
        prompt_ += f"{idx + 1}. 输入:\n{input}\n"
        prompt_ += f"{idx + 1}. 输出:\n{output}\n"
    prompt_ += f"###\n"

    message = [{"role": "assistant", "content": prompt_}]
    completion = openai.ChatCompletion.create(
        model= "gpt-3.5-turbo",
        messages= message,
        temperature= 1.0,
        max_tokens= 500,
        top_p= 1.0,
        frequency_penalty= 0.0,
        presence_penalty= 0.0
    )
    response = completion.choices[0].message["content"]
    print(response)
    response = parse_response(response)

    time.sleep(1)
    lock.acquire()
    if response:
        generate_task.append(response)
        bar.next()
        if len(generate_task)%40 == 0:
            with open("./dialogue_task.json", "w", encoding="utf-8") as f:
                json.dump(generate_task, f, indent=4, ensure_ascii=False)
    lock.release()
   
def main():
    with open("./dialogue_seed_task.json", "r") as f:
        seed_task_list = json.load(f)

    ### 需要提供你的openai api key
    api_keys=[]

    prompt = f"你被要求提供7个多样化的任务指令。这些任务指令将被提供给GPT模型，我们将评估GPT模型完成指令的能力。\n \
    以下是你提供指令需要满足的要求：\n \
    1.指令用中文书写，指令应该是一个医疗任务。\n \
    2.指令类型应该是多样化的，包括各种类型的任务，类别种类例如：病情诊断，病因分析，病理诊断，治疗方案，就医建议，指标解读，药物剂量，用药建议，医疗建议，医学知识，疾病描述，后果表述，注意事项，功效作用，医疗费用，预防措施，预后评估，其他\n \
    3.你应该给指令生成适当的输入，输入字段应包含为指令提供的具体示例，它应该是一个医疗问题，含有有用的医学信息，例如病灶描述，体检指标数值，药物剂量等，不应包含简单的占位符。输入应提供充实的内容，使指令具有挑战性。\n \
    4.输出应该是针对指令和输入的恰当回答，如果输入的信息不足以进行判断需要进一步询问。\n \
    5.输入输出相关的疾病应该是多样化的，包含各种类型的疾病和药品信息。\
    下面是7个任务指令的列表： \n"

    print('here reading history result')
    if os.path.exists('./dialogue_task.json'):
        with open("./dialogue_task.json", "r") as f:
            generate_task = json.load(f)
        # with open("./dialogue_task_intermediate.json", "w", encoding="utf-8") as f:
        #     json.dump(generate_task, f, indent=4, ensure_ascii=False)
    else:
        generate_task=[]
    lock=Lock()
    global count
    count=0

    ###  需要生成的medial dialogue数量
    require_length=52000

    data_list = [[i] for i in range(require_length-len(generate_task))]
    data_list = choices(data_list,k=10)
    print('bar here')
    bar = Bar('Processing', max=len(data_list),suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')

    print('building threads')
    pool = ThreadPool(processes=20)

    res = pool.starmap(gpt_generate, [[i] for i in data_list])
    pool.close()
    pool.join()
    bar.finish()
    print('save all')
    with open("./dialogue_task.json", "w", encoding="utf-8") as f:
        json.dump(generate_task, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()