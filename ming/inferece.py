from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

torch.set_default_device("cuda")
model_path = "Qwen/Qwen1.5-1.8B"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                                        use_fast=False)

tokenizer.pad_token_id = 151643
inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
