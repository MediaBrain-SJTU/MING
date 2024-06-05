import json
import os
import openai
from openai import OpenAI
import pdb
import re
import numpy as np
from tqdm import tqdm
from progress.bar import Bar
from random import sample
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

TEMPLATE="""
You are an AI evaluator specializing in assessing the quality of answers
provided by other language models . Your primary goal is to rate the
answers based on their fluency , relevance , completeness , proficiency
in medicine . Use the following scales to evaluate each criterion :
Fluency :
1: Completely broken and unreadable sentence pieces
2: Mostly broken with few readable tokens
3: Moderately fluent but with limited vocabulary
4: Mostly coherent in expressing complex subjects
5: Human - level fluency
Relevance :
1: Completely unrelated to the question
2: Some relation to the question , but mostly off - topic
3: Relevant , but lacking focus or key details
4: Highly relevant , addressing the main aspects of the question
5: Directly relevant and precisely targeted to the question
Completeness :
1: Extremely incomplete
2: Almost incomplete with limited information
3: Moderate completeness with some information
4: Mostly complete with most of the information displayed
5: Fully complete with all information presented
Proficiency in medicine :
1: Using plain languages with no medical terminology .
2: Equipped with some medical knowledge but lacking in - depth details
3: Conveying moderately complex medical information with clarity
4: Showing solid grasp of medical terminology but having some minor
mistakes in detail
5: Fully correct in all presented medical knowledge
You will be provided with the following information :
- a description
- a question based on the description and conversation
- the solution to the question
- a model' s answer to the question
[ description ]
{description}
[ end of description ]
[ question ]
{question}
[ end of question ]
[ solution ]
{solution}
[ end of solution ]
[ answer ]
{answer}
[ end of answer ]
Make sure to provide your evaluation results in JSON format and ONLY the
JSON , with separate ratings for each of the mentioned criteria as in
the following example :
{{"fluency": 3, "relevance": 3, "completeness": 3, "proficiency": 3}}
"""

def score_fn(question, bar, model="gpt-4o"):
    data = question[0]
    
    prompt = TEMPLATE.format(
        description = data["additional_info"]["description"],
        question = data["additional_info"]["question"],
        solution = data["additional_info"]["solution"],
        answer = data["text"])
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=40
    )

    score = completion.choices[0].message.content
    
    if "json\n" in score:
        score = score.strip("`").replace("json\n", "")
    try:
        data["score"] = json.loads(score)
    except:
        print(f"[Warning]: the output '{score}' can not be converted to the json formate!")
        data["score"] = score

    bar.next() 


def cmb_eval(args, questions):
    # output_path = args.output_file.rsplit("/", 1)[0]
    # if os.path.exist(f"{output_path}/gpt4_score.json"):
    #     with open(f"{output_path}/gpt4_score.json", "r") as f:
    #         datas = json.load(f)  
    output_path = args.output_file.rsplit("/", 1)[0]
    if not os.path.exists(f"{output_path}/gpt4_eval_results.json"):
        print('bar here')
        bar = Bar(f'Processing', max=len(questions), suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')
        # df.text.apply(lambda x: get_embedding(x, model=embedding_model))
        pool = ThreadPool(processes=20)
        res = pool.starmap(score_fn, [[question, bar] for question in zip(questions)])
        pool.close()
        pool.join()
        bar.finish()

        with open(f"{output_path}/gpt4_eval_results.json", "w") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)
    else:
        with open(f"{output_path}/gpt4_eval_results.json", "r") as f:
            questions = json.load(f)


    if "type" not in questions[0]["additional_info"]:
        avg_score = {'fluency': [], 'relevance': [], 'completeness': [], 'proficiency': []}
        for dimension in avg_score:
            avg_score[dimension] = sum([question["score"][dimension] for question in questions]) / len(questions)
    else:
        types = list(set(question["additional_info"]["type"] for question in questions))
        avg_score = {'type': types, 'fluency': [], 'relevance': [], 'completeness': [], 'proficiency': []}
        for task_type in types:
            for dimension in ['fluency', 'relevance', 'completeness', 'proficiency']:
                scores = [question["score"][dimension] for question in questions if question["additional_info"]["type"] == task_type]
                avg_score[dimension].append(sum(scores) / len(scores))

        print(avg_score)
        df = pd.DataFrame(avg_score)
        df.to_csv(f"{output_path}/score.csv")
    

    
    return avg_score, questions
