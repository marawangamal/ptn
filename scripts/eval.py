import os
import lm_eval
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

from mtp.mthf.modelling_mthf import MultiTokenHF, MultiTokenHFConfig


# Create model and tokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct") # hf model
model = MultiTokenHF(
    MultiTokenHFConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        model_head="multihead",
        horizon=2,
        loss_type="joint",
        pretrained=True,
    )
)  # mthf model

# Move to GPU
model.to("cuda")
model.train()


# Sample from model
output = model.generate(
    tok.encode("Hello, I'm a language model,", return_tensors="pt").to(model.device),
    max_new_tokens=64,
    do_sample=False,
)
print(tok.decode(output[0]))

# Custom
# script_dir = os.path.dirname(os.path.abspath(__file__))
# task_manager = lm_eval.tasks.TaskManager(
#     include_path=os.path.join(script_dir, "configs")
# )

# Evaluate
output = simple_evaluate(
    model=lm_eval.models.huggingface.HFLM(pretrained=model),
    tasks=["gsm8k_cot"],
    batch_size=32,
    limit=32,
)

for task_name in output["results"].keys():
    for metric_name in output["results"][task_name].keys():
        print(
            f"{task_name}/{metric_name} = {output['results'][task_name][metric_name]}"
        )
