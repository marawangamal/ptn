import os
import lm_eval
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model.to("cuda")

# Custom
# script_dir = os.path.dirname(os.path.abspath(__file__))
# task_manager = lm_eval.tasks.TaskManager(
#     include_path=os.path.join(script_dir, "configs")
# )
output = simple_evaluate(
    model=lm_eval.models.huggingface.HFLM(pretrained=model),
    tasks=["gsm8k_cot"],
    batch_size=32,
    cache_prompts=True,
    device="cuda",
    dtype="bfloat16",
)
print(output)
