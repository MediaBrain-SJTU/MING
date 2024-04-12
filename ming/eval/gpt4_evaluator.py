import argparse
import json
import os

import openai
import tqdm
import ray
import time
import jsonlines 

NUM_SECONDS_TO_SLEEP = 1

def get_rule():
    return {
        'prompt': '''
请根据给定的标准答案和解释评估答案的质量。你应该分别给出学生回答的两个分数。第一个分数是对答案正确性的评分，第二个分数是对解释质量的评分。分数范围应该是0到5，其中5分为最佳。如果回答推断出了正确答案，第一个分数应该是5分。如果回答不能推断出正确答案，两个分数都应该是0分。第二个分数基于回答与给定解释的相关性给出。相关性越高，分数越高。请在第一行分别给出两个分数，并用空格分隔。''',
        'role': '学生',
    }
from openai import OpenAI



@ray.remote(num_cpus=4)
def get_eval(content: str, max_tokens: int):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                    {"role": "user", "content": content}
                ],
                temperature=0.2,
                max_tokens=max_tokens
            )
            
            break
        except openai.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    print('success!')
    return completion.choices[0].message.content


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-a', '--answer_list')
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    ray.init()

    # load args.answer_list, which is jsonl file
    question_lines = []
    with open(os.path.expanduser(args.answer_list), 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            question_lines.append(item)
    # ques_js = os.path.expanduser(args.answer_list)
    # f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    # f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    # rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    review_file = open(f'{args.output}', 'w')

    js_list = []
    handles = []
    idx = 0
    # for ques_js in zip(f_q, f_ans1,):
        # if idx == 1:
        #     break

    # ques = json.loads(ques_js)
    # ans1 = json.loads(ans1_js)
    # ans2 = json.loads(ans2_js)

    # difficulty_level = json.loads(ques_js)['difficulty_level']
    # if category in rule_dict:
    #     rule = rule_dict[category]
    # else:
    #     rule = rule_dict['default']
    # prompt = rule['prompt']
    # role = rule['role']
    rule = get_rule()
    role = rule['role']
    prompt = rule['prompt']
    for ques in question_lines:
        difficulty_level = ques['difficulty_level']
        content = (f'[问题]\n{ques["prompt"]}\n\n'
                    f'[{role}回答]\n{ques["text"]}\n[{role}回答结束]\n\n'
                    f'[标准答案]\n{ques["answer"]}\n[标准答案结束]\n\n'
                    f'[标准答案解释]\n{ques["explanation"]}\n[标准答案解释结束]\n\n'
                    f'[系统]\n{prompt}\n\n')
        js_list.append({
            'id': idx+1,
            'question': ques['prompt'],
            'answer': ques['answer'],
            'difficlty_level': difficulty_level})
        idx += 1
        handles.append(get_eval.remote(content, args.max_tokens))
        # To avoid the rate limit set by OpenAI
        # time.sleep(NUM_SECONDS_TO_SLEEP)

    reviews = ray.get(handles)
    total_questions = len(question_lines)
    acc_score_avg = exp_score_avg = 0
    for idx, review in enumerate(reviews):
        scores = parse_score(review)
        js_list[idx]['content'] = review
        js_list[idx]['tuple'] = scores
        acc_score, exp_score = scores
        acc_score_avg += acc_score 
        exp_score_avg += exp_score
        review_file.write(json.dumps(js_list[idx], ensure_ascii=False) + '\n')
        review_file.flush()
    review_file.close()
    
    acc_score_avg /= total_questions
    exp_score_avg /= total_questions
    print(f'Average accuracy score: {acc_score_avg}')
    print(f'Average explanation score: {exp_score_avg}')