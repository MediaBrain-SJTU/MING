import os 
import json 
import jsonlines

import argparse 
from tqdm import tqdm, trange
from subprocess import PIPE, Popen, TimeoutExpired
import tempfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # input_file is a jsonl file with the following format:
    questions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    
    total_num = len(questions)
    total_score = 0
    for line in tqdm(questions, total=total_num):
        text = line['text']
        function_name = line['function_name']
        additional_info = line['additional_info']
        function_name = additional_info['function_name']
        test = additional_info['test']
        executable_code = function_name + text
        if isinstance(line['test'], str):
            test_code = executable_code + "\n" + test
        else:
            test_code = executable_code + "\n" + "\n".join(test)
        
        
        ff = tempfile.NamedTemporaryFile()
        ff.write(test_code)
        name = ff.name 
        p = Popen(f'python {name}', shell=True, stdout=PIPE, stderr=PIPE)
        time_limit = 15  # seconds
        scores = 1
        try:
            stdout, stderr = p.communicate(timeout=time_limit)
        except TimeoutExpired:
            # Linux
            # os.killpg(p.pid, signal.SIGTERM)
            # Windows
            Popen("TASKKILL /F /PID {pid} /T".format(pid=p.pid))
            scores = 0
        if stderr:
            scores = 0
        
        total_score += scores 
        ff.close()
    
    pass_at_1 = total_score / total_num
    print(f"Pass@1: {pass_at_1}")