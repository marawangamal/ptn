# fineweb_10b_tokenize.py
import argparse
import os
import re
from datatrove.executor import (
    LocalPipelineExecutor,
)  # or SlurmPipelineExecutor / RayPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import DocumentTokenizer
from transformers import AutoTokenizer


def normalize_str(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ------------------------------------------------------------------
    # 1) Reader – pulls data lazily from the Hub (no local download)
    # ------------------------------------------------------------------
    fw_reader = HuggingFaceDatasetReader(
        dataset="HuggingFaceFW/fineweb",
        dataset_options={
            "split": "train",
            "name": "sample-10BT",  # ≃10 billion GPT2 tokens (≈27GB parquet)
            "cache_dir": f"{os.environ['SCRATCH']}/huggingface/datasets",  # Explicit cache
            "download_mode": "reuse_cache_if_exists",  # Don't re-download
        },
        text_key="text",  # column to feed the tokenizer
        id_key="id",  # (optional) keeps the UUID in metadata
    )

    # ------------------------------------------------------------------
    # 2) Tokeniser – writes .ds shards ready for Nanotron / Nanoset
    # ------------------------------------------------------------------
    tok_block = DocumentTokenizer(
        output_folder=f"./fineweb-10bt-ds-{normalize_str(args.tokenizer)}",  # local path also works
        tokenizer_name_or_path=args.tokenizer,
        eos_token=tokenizer.eos_token,  # or tok.eos_token
        batch_size=10_000,  # how many docs per tokenizer batch
        max_tokens_per_file=int(1e8),  # shard size (~400 MB with 4byte ints)
        # pack_sequences=True,  # turn docs into contiguous 2k/4k ctx windows
        shuffle_documents=True,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 3) Launch – scale locally or on your cluster
    # ------------------------------------------------------------------
    n_cpus = int(os.environ.get("SLURM_NPROCS", 1))
    job = LocalPipelineExecutor(  # drop‑in SlurmPipelineExecutor if you’re on HPC
        pipeline=[fw_reader, tok_block],
        tasks=n_cpus,
        workers=max(1, n_cpus // 2),
        logging_dir=f"./logs/fineweb10b-{normalize_str(args.tokenizer)}",
    )
    job.run()

    # ------------------------------------------------------------------
    # EXAMPLE USAGE DURING TRAINING
    #  from datatrove.utils.dataset import DatatroveFolderDataset
    #  ds = DatatroveFolderDataset(ds_path, seq_len=1024)
