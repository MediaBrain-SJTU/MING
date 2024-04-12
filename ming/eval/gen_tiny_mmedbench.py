import pandas as pd 
import argparse
import json
import os 

def extract_tiny_with_class(df, class_name, ratio=0.1):
    # 确定每个类别的唯一值和它们的分布
    class_counts = df[class_name].value_counts(normalize=True)
    
    # 计算每个类别应抽取的样本数
    samples_per_class = (class_counts * len(df) * ratio).round().astype(int)
    
    # 确保每个类别至少有一个样本
    samples_per_class = samples_per_class.apply(lambda x: max(x, 1))
    
    # 初始化一个空的DataFrame用于收集抽样结果
    sampled_df = pd.DataFrame(columns=df.columns)
    
    # 对每个类别进行抽样
    for class_value, n_samples in samples_per_class.items():
        # 对当前类别进行抽样
        sampled_class_df = df[df[class_name] == class_value].sample(n=n_samples)
        # 将抽样结果添加到收集用的DataFrame
        sampled_df = pd.concat([sampled_df, sampled_class_df], ignore_index=True)
    
    return sampled_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tiny cmexam dataset')
    parser.add_argument('--input', type=str, help='Input jsonl file')
    parser.add_argument('--output', type=str, help='Output jsonl file')
    parser.add_argument("--extract_key", type=str, help='based on what to extract, default to answer_idx', default="answer_idx")
    parser.add_argument('--ratio', type=float, help='ratio of the dataset to be extracted', default=0.1)
    args = parser.parse_args()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.input), "r")]
    # convert dict file to pd.DataFrame
    df = pd.DataFrame(questions)
    
    # first remove those items whose explanation is empty
    df = df[df['rationale'].notna()]
    df = extract_tiny_with_class(df, args.extract_key, args.ratio)
    
    dict_file = df.to_dict(orient='records')
    # save to jsonl file
    with open(os.path.expanduser(args.output), "w") as f:
        for item in dict_file:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")