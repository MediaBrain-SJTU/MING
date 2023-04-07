## 注意事项
- peft 需要安装指定版本，可以安装本地版本（peft-main）
  - 同时要根据自己模型的不同特点选择不同的target_modules=[xxx]
  - modules_to_save：除了lora部分之外，还有哪些层可以被训练，并且需要保存
- 模型中的get_position_ids 需要改成 **context_length = len(seq)**
- 模型中的padding与以往方式不同
  - 原有格式为 [que] + [reply]
  - 需要改为 [que + [tokenizer.mask_token_id]] + [pad_token]*n + [ [tokenizer.bos_token_id]+ reply + [tokenizer.eop_token_id]]
- 模型中只允许bs为1进行训练，如果想多bs需要自行修改代码，已修改好
## 各个文件
- train.py：为lora微调
- demo.py：为demo代码
## 配置环境
- pip install -r requirements.txt
## 训练步骤
- 请提前下载好chatglm-6b权重，参考[https://huggingface.co/THUDM/chatglm-6b]，放在model文件夹下 ！函数不要进行修改
- 运行train.py即可
  - 可以指定多gpu进行模型并行，可以根据模型大小设置不同的batch_size，但是无法进行数据并行
  - 参考：8卡batch_size为4，每张卡10G
## 测试步骤
- 首先按照配置环境步骤配置环境
- 运行demo.py即可 ！注意：需要使用gpu
  - 各个参数含义
    - model_path：原始chatglm模型的存放地址，./model
    - peft_path：lora微调后的模型存放地址，./Fine_Tuning_Results/lora_2e-5/lora.p
    - gpu_id:全部能使用的gpu_id，默认为'0'，可以设置不同的gpu'0,1,2,3,4,5,6,7'
  - 输入，默认模型不具备记忆能力，如果想进行多轮对话，需要手动输入history
    - 不进行多轮对话，直接输入问题：Q
    - 进行多轮对话，具体输入形式为：Q1[\n]A1[\n]Q2[\n]A2[\n]Q3 !使用 **[\n]** 对问题和答案进行分割
- 提示：
  - 半精度单卡显存:lora:15G
