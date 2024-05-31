from safetensors.torch import load_file 
import torch 
import os 

data_path = "/mnt/hwfile/medai/jiangshuyang.p/checkpoints/cblue_16k-qwen1.5-1.8b-molora-r16a32_share_expert_2_orthlora"

lora_state_dict = load_file(os.path.join(data_path, "model.safetensors"))
for k, v in lora_state_dict.items():
    print(k, v.shape)
print("*" * 100)

nonlora_state_dict = torch.load(os.path.join(data_path, "non_lora_trainables.bin"))
for k, v in nonlora_state_dict.items():
    print(k, v.shape)
