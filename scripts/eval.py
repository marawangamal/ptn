import lm_eval
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

output = simple_evaluate(
    model=lm_eval.models.huggingface.HFLM(pretrained=model),
    tasks=["gsm8k_cot_llama"],
    # limit
    limit=10,
    batch_size=8,
)
print(output["results"])
