import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from huggingface_hub import HfApi, create_repo
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput


# -----------------------------
# Tiny custom model definition
# -----------------------------


class TinyMLPConfig(PretrainedConfig):
    model_type = "tinymlp"

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 2,
        max_position_embeddings: int = 128,
        layer_norm_epsilon: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_epsilon


class TinyMLPModel(PreTrainedModel):
    config_class = TinyMLPConfig

    def __init__(self, config: TinyMLPConfig):
        super().__init__(config)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = torch.nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        layers = []
        for _ in range(config.num_hidden_layers):
            layers.extend(
                [
                    torch.nn.Linear(config.hidden_size, config.intermediate_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(config.intermediate_size, config.hidden_size),
                    torch.nn.LayerNorm(
                        config.hidden_size, eps=config.layer_norm_epsilon
                    ),
                ]
            )
        self.mlp = torch.nn.Sequential(*layers)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)

        if attention_mask is not None:
            # Expand mask to (batch, seq, 1) and zero-out padded tokens
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).type_as(
                hidden_states
            )

        hidden_states = self.mlp(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings


# -----------------------------
# Utilities to save/push model
# -----------------------------


def write_remote_code_file(target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    code_path = os.path.join(target_dir, "modeling_tinymlp.py")
    code_contents = (
        "import torch\n"
        "from transformers import PreTrainedModel, PretrainedConfig\n"
        "from transformers.modeling_outputs import BaseModelOutput\n\n"
        "class TinyMLPConfig(PretrainedConfig):\n"
        "    model_type = 'tinymlp'\n\n"
        "    def __init__(self, vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2, max_position_embeddings=128, layer_norm_epsilon=1e-5, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.vocab_size = vocab_size\n"
        "        self.hidden_size = hidden_size\n"
        "        self.intermediate_size = intermediate_size\n"
        "        self.num_hidden_layers = num_hidden_layers\n"
        "        self.max_position_embeddings = max_position_embeddings\n"
        "        self.layer_norm_epsilon = layer_norm_epsilon\n\n"
        "class TinyMLPModel(PreTrainedModel):\n"
        "    config_class = TinyMLPConfig\n\n"
        "    def __init__(self, config: TinyMLPConfig):\n"
        "        super().__init__(config)\n"
        "        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)\n"
        "        self.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)\n"
        "        layers = []\n"
        "        for _ in range(config.num_hidden_layers):\n"
        "            layers.extend([\n"
        "                torch.nn.Linear(config.hidden_size, config.intermediate_size),\n"
        "                torch.nn.ReLU(),\n"
        "                torch.nn.Linear(config.intermediate_size, config.hidden_size),\n"
        "                torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon),\n"
        "            ])\n"
        "        self.mlp = torch.nn.Sequential(*layers)\n"
        "        self.post_init()\n\n"
        "    def forward(self, input_ids: torch.LongTensor, attention_mask=None, **kwargs) -> BaseModelOutput:\n"
        "        device = input_ids.device\n"
        "        batch_size, seq_len = input_ids.shape\n"
        "        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)\n"
        "        positions = positions.unsqueeze(0).expand(batch_size, seq_len)\n"
        "        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)\n"
        "        if attention_mask is not None:\n"
        "            hidden_states = hidden_states * attention_mask.unsqueeze(-1).type_as(hidden_states)\n"
        "        hidden_states = self.mlp(hidden_states)\n"
        "        return BaseModelOutput(last_hidden_state=hidden_states)\n"
        "    def get_input_embeddings(self):\n"
        "        return self.embed_tokens\n"
        "    def set_input_embeddings(self, new_embeddings):\n"
        "        self.embed_tokens = new_embeddings\n"
    )
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code_contents)


def write_model_card(target_dir: str, repo_id: str) -> None:
    readme_path = os.path.join(target_dir, "README.md")
    content = f"""
# TinyMLP: a minimal custom Transformers model

This repository contains a minimal MLP model demonstrating how to host custom architectures on the Hugging Face Hub using `trust_remote_code=True`.

## Usage

```python
from transformers import AutoConfig, AutoModel

model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)
config = AutoConfig.from_pretrained("{repo_id}", trust_remote_code=True)

print(model.__class__.__name__, config.model_type)
```

The model exposes a standard `last_hidden_state` in its outputs.
""".strip()
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def save_local_minimal_model(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    config = TinyMLPConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        max_position_embeddings=128,
    )

    model = TinyMLPModel(config)

    # Save model + config
    model.save_pretrained(output_dir)

    # Save the remote code file and set up auto_map for trust_remote_code
    write_remote_code_file(output_dir)

    # Update config.json with auto_map so Auto classes work with trust_remote_code
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["auto_map"] = {
        "AutoConfig": "modeling_tinymlp.TinyMLPConfig",
        "AutoModel": "modeling_tinymlp.TinyMLPModel",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def push_folder_to_hub(
    local_dir: str, repo_id: str, private: bool, commit_message: str
) -> None:
    create_repo(repo_id=repo_id, exist_ok=True, private=private)
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message=commit_message,
        ignore_patterns=["*.pt", "*.bin.tmp", "tmp/*"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create and push a minimal custom Transformers model to the Hub"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repo on the Hub, e.g., username/tinymlp",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/tinymlp",
        help="Local directory to save the model",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repo on the Hub"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Only save locally; do not push to the Hub",
    )
    parser.add_argument("--commit-message", default="Add TinyMLP minimal custom model")
    args = parser.parse_args()

    save_local_minimal_model(args.output_dir)
    write_model_card(args.output_dir, args.repo_id)

    if not args.no_push:
        push_folder_to_hub(
            args.output_dir, args.repo_id, args.private, args.commit_message
        )
        print(f"Pushed to https://huggingface.co/{args.repo_id}")
    else:
        print(f"Saved locally at: {args.output_dir}")

    # Quick local verification
    device = torch.device("cpu")
    model = TinyMLPModel.from_pretrained(args.output_dir).to(device)
    input_ids = torch.randint(
        0, model.config.vocab_size, (2, 8), dtype=torch.long, device=device
    )
    outputs = model(input_ids)
    print("Local forward OK. Output shape:", tuple(outputs.last_hidden_state.shape))


if __name__ == "__main__":
    main()
