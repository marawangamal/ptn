import os
import lm_eval
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# move to GPU
model.to("cuda")

# Custom
# script_dir = os.path.dirname(os.path.abspath(__file__))
# task_manager = lm_eval.tasks.TaskManager(
#     include_path=os.path.join(script_dir, "configs")
# )
output = simple_evaluate(
    model=lm_eval.models.huggingface.HFLM(pretrained=model),
    tasks=["gsm8k_cot"],
    # limit
    limit=10,
    batch_size=8,
    # task_manager=task_manager,
)
print(output)
