from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen2.5-14B_model_path", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "output_Qwen_2.5-14B_output")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("Qwen2.5_output_merged_14B", max_shard_size="4096MB", safe_serialization=True)





tokenizer = AutoTokenizer.from_pretrained(
    "Qwen2.5-14B_model_path",
    trust_remote_code=True
)

tokenizer.save_pretrained("Qwen2.5_output_merged_14B")