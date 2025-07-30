# fineweb_10b_tokenize.py
from datatrove.executor import (
    LocalPipelineExecutor,
)  # or SlurmPipelineExecutor / RayPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import DocumentTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")


# ------------------------------------------------------------------
# 1) Reader – pulls data lazily from the Hub (no local download)
# ------------------------------------------------------------------
fw_reader = HuggingFaceDatasetReader(
    dataset="HuggingFaceFW/fineweb",
    dataset_options={
        "split": "train",
        "name": "sample-10BT",  # ≃10 billion GPT‑2 tokens (≈27 GB parquet) :contentReference[oaicite:0]{index=0}
    },
    text_key="text",  # column to feed the tokenizer
    id_key="id",  # (optional) keeps the UUID in metadata
)

# ------------------------------------------------------------------
# 2) Tokeniser – writes .ds shards ready for Nanotron / Nanoset
# ------------------------------------------------------------------
tok_block = DocumentTokenizer(
    output_folder="./fineweb-10bt-ds",  # local path also works
    tokenizer_name_or_path="gpt2",  # <-- plug your own tokenizer here
    eos_token=tokenizer.eos_token,  # or tok.eos_token
    batch_size=10_000,  # how many docs per tokenizer batch
    max_tokens_per_file=int(1e8),  # shard size (~400 MB with 4‑byte ints)
    # pack_sequences=True,  # turn docs into contiguous 2 k/4 k ctx windows
    shuffle_documents=True,
    seed=42,
)

# ------------------------------------------------------------------
# 3) Launch – scale locally or on your cluster
# ------------------------------------------------------------------
job = LocalPipelineExecutor(  # drop‑in SlurmPipelineExecutor if you’re on HPC
    pipeline=[fw_reader, tok_block],
    tasks=1,  # 128 parallel processes -> adjust to CPU quota
    workers=1,  # how many run concurrently
    logging_dir="./logs/fineweb10b",
)
job.run()
