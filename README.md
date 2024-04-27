

# 明医 (MING)：中文医疗问诊大模型

<p align="center">
  <img src=".\img\bgimage.png" width=800px/>
</p>


<div align="center"><img src="https://img.shields.io/badge/Version-1.3--alpha-brightgreen"> <img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg"> <img src="https://img.shields.io/badge/python-3.9+-blue.svg"></div>

## 🌐项目简介

本项目开源了基于医疗指令微调的中文医疗问诊模型：**明医 (MING)**。目前模型的主要功能如下：

<!DOCTYPE html>
<html>
<body>
<table style="width: 100%;">
  <tr style="border-collapse: collapse; border: transparent;">
      <td style="width: 50%; border-collapse: collapse;border: transparent;"><img src=".\img\demo1.gif" alt="demo1"/></td>
      <td style="width: 50%; border-collapse: collapse;border: transparent;"><img src=".\img\demo2.gif" alt="demo2"/></td>
  </tr>
  <tr style="border-collapse: collapse; border: transparent;">
      <td style="width: 50%; border-collapse: collapse;border: transparent;" ><div align="center"><strong>医疗问答</strong>：对医疗问题进行解答，对案例进行分析。</div></td>
      <td style="width: 50%; border-collapse: collapse;border: transparent;"><div align="center"><strong>智能问诊</strong>：多轮问诊后给出诊断结果和建议。</div></td>
  </tr>
</table>
</body>
</html>

## 📄相关论文
* MING-MOE技术报告

  [MING-MOE: Enhancing Medical Multi-Task Learning in Large Language Models with Sparse Mixture of Low-Rank Adapter Experts](https://arxiv.org/pdf/2404.09027.pdf)

* 基于多智能体交互的大语言模型多轮问诊自动评估框架

  [Automatic Interactive Evaluation for Large Language Models with State Aware Patient Simulator](https://arxiv.org/pdf/2403.08495.pdf)

## 💫更新
* 🔥 [2024/04/14] 开源了基于Qwen1.5指令微调的专家混合模型MING-MOE

* [2024/03/14] 开源了基于Qwen1.5-1.8b指令微调的MING-1.8B

* [2023/07/25] 开源了基于bloomz-7b指令微调的MING-7B

* [2023/07/25] MedicalGPT-zh更名为**MING**

  

##  🔬开源模型

<!DOCTYPE html>
<html>
<head>
</head>
<body>
<table style="width: 80%;">
  <tr>
      <td style="width: 20%;"><div align="center"><strong>模型</strong></div></td>
      <td style="width: 20%;"><div align="center"><strong>基座</strong></div></td>
      <td style="width: 30%;"><div align="center"><strong>HuggingFace</strong></div></td>
  </tr>
  
  <tr>
      <td><center>MING-7B</center></td>
      <td><center><a href="https://huggingface.co/bigscience/bloomz-7b1-mt">bloomz-7b1-mt</a></center></td>
      <td><center>🤗<a href="https://huggingface.co/BlueZeros/MING-7B">MING-7B</a></center></td>
  </tr>

  <tr>
      <td><center>MING-1.8B</center></td>
      <td><center><a href="https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat">Qwen1.5-1.8B</a></center></td>
      <td><center>🤗<a href="https://huggingface.co/BlueZeros/MING-1.8B">MING-1.8B</a></center></td>
  </tr>

  <tr>
      <td><center>MING-MOE-1.8B</center></td>
      <td><center><a href="https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat">Qwen1.5-1.8B</a></center></td>
      <td><center>🤗<a href="https://huggingface.co/BlueZeros/MING-MOE-1.8B">MING-MOE-1.8B</a></center></td>
  </tr>

  <tr>
      <td><center>MING-MOE-4B</center></td>
      <td><center><a href="https://huggingface.co/Qwen/Qwen1.5-4B-Chat">Qwen1.5-4B</a></center></td>
      <td><center>🤗<a href="https://huggingface.co/BlueZeros/MING-MOE-4B">MING-MOE-4B</a></center></td>
  </tr>

  <tr>
      <td><center>MING-MOE-7B</center></td>
      <td><center><a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat">Qwen1.5-7B</a></center></td>
      <td><center>🤗<a href="https://huggingface.co/BlueZeros/MING-MOE-7B">MING-MOE-7B</a></center></td>
  </tr>

  <tr>
      <td><center>MING-MOE-14B</center></td>
      <td><center><a href="https://huggingface.co/Qwen/Qwen1.5-14B-Chat">Qwen1.5-14B</a></center></td>
      <td><center>🤗<a href="https://huggingface.co/BlueZeros/MING-MOE-14B">MING-MOE-14B</a></center></td>
  </tr>
</table>
</body>
</html>


## ⚡快速开始

1. 配置环境（测试环境如下，具体版本可以根据实际需求配置）

   * python==3.9.16
   * pytorch==2.0.1+cu117
   * peft==0.9.0

2. 安装项目依赖 

   ```bash
   git clone https://github.com/MediaBrain-SJTU/MING
   cd MING
   pip install -e .
   ```

2. 下载模型参数并运行（要求单卡显存 >= 15G）
    * MING-MOE
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.cli \
       --model_path {path_to_checkpoint} \ # 模型路径
       --model_base {path_to_base_model} \ # 基座模型路径
       --max-new-token 3072 # 输出最大长度
   ```

   * MING-1.8B
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.cli \
       --model_path {path_to_checkpoint} \ # 模型路径
       --max-new-token 2048 # 输出最大长度
   ```

   * MING-7B
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.cli \
       --model-path {path_to_checkpoint} \ # 模型路径
       --conv_template bloom \ # prompt
       --max-new-token 512 \ # 输出最大长度
       --beam-size 3 \ # beam search宽度
       --temperature 1.2 # 采样温度
   ```
   
   * 注：由于transformers库的问题，当beam-size > 1时，需要满足temperature>=1.0，否则会报错。

4. 命令行运行实例

   * 对话支持多轮

   * 对话中输入关键词 `new chat` 能够开启新一轮对话。


## 🧭测试样例
<p align="center">
  <img src=".\img\case1.png" width=800px/>
</p>

<p align="center">
  <img src=".\img\case2.png" width=800px/>
</p>



## 🪶贡献

本项目由上海交通大学未来媒体网络协同创新中心和上海人工智能实验室智慧医疗中心合作研发。模型数据系统主要由廖育生，江书洋，刘泓呈，孟昱同完成，指导教师为[王钰](https://yuwangsjtu.github.io)副教授。



## 免责声明

预训练模型是基于大量语料库和算法模型进行训练的，并且在训练过程中可能存在偏差、错误和不完整的信息。因此，本项目提供的预训练模型仅供参考和研究使用，并不能保证其准确性和可靠性。使用预训练模型产生的结果可能存在误差和偏差，不能用于实际应用或决策。本项目不对使用预训练模型所产生的结果承担任何责任，也不对因使用预训练模型所产生的任何损失承担责任。使用者在使用预训练模型时应自行承担风险并进行自我验证。



## 引用

如果你使用了本项目的数据或者代码，请声明引用

```latex
@misc{liao2024mingmoe,
      title={MING-MOE: Enhancing Medical Multi-Task Learning in Large Language Models with Sparse Mixture of Low-Rank Adapter Experts}, 
      author={Yusheng Liao and Shuyang Jiang and Yu Wang and Yanfeng Wang},
      year={2024},
      eprint={2404.09027},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MediaBrain-SJTU/MING&type=Date)](https://star-history.com/#MediaBrain-SJTU/MING&Date)

