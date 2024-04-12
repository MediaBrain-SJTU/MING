import jsonlines
import json
import argparse
import os 
from tqdm import tqdm
import re 

def parse_score(predict: str, answer: str):
    answer_candidates = set(["A", "B", "C", "D", "E"])
    if predict in answer_candidates:
        return 1 if predict == answer else 0 
    else:
        # use re to verdict whether any capital character in predict
        parsed_predict = re.findall(r'[A-E]', predict)[0]
        return 1 if parsed_predict == answer else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-a', '--answer_list')
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    
    question_lines = []
    with open(os.path.expanduser(args.answer_list), 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            question_lines.append(item)
            
    review_file = open(f'{args.output}', 'w')

    js_list = []
    handles = []
    idx = 0

    acc_score_avg = 0
    total_questions = len(question_lines)
    for ques in tqdm(question_lines):

        score = parse_score(ques['text'], ques['answer'])
        temp_save_result = {
            'id': idx+1,
            'question': ques['prompt'],
            'answer': ques['answer'],
            'score': score
            }
        acc_score_avg += score
        review_file.write(json.dumps(temp_save_result, ensure_ascii=False) + '\n')
        review_file.flush()
    
    review_file.close()


    
    acc_score_avg /= total_questions

    print(f'Average accuracy score: {acc_score_avg}')
