import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnsight import LanguageModel
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import wandb
import glob
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

cache_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf_cache"
)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

TAILS = {
    "socioeco": "### Assistant: I think the user's socioeconomic status is",
    "religion": "### Assistant: I think the user's religion is",
    "location": "### Assistant: I think the user's location is",
}


def truncate_to_tail(text: str, tail: str) -> str:
    text = text.replace("\r\n", "\n")
    idx = text.rfind(tail)
    if idx == -1:
        return text
    return text[: idx + len(tail)]


def sanitize_conversation(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").replace("\r\n", "\n").split("\n")]
    kept = [
        ln
        for ln in lines
        if ln.startswith("### Human:") or ln.startswith("### Assistant:")
    ]
    return "\n".join(kept).strip() or text


@dataclass
class ProbeConfig:
    steps: int = 50
    layer: int = 31
    use_wandb: bool = True
    wandb_project: str = "user-models"
    model: LanguageModel = field(
        default_factory=lambda: LanguageModel(
            "meta-llama/Meta-Llama-3.1-8B", device_map="auto"
        )
    )
    data_dir: str = "data"
    batch_size: int = 32
    learning_rate: float = 1e-3
    test_size: float = 0.2
    random_seed: int = 42
    task_prefix: str = "socioeco"  # or 'religion', 'location'


device = "cuda" if torch.cuda.is_available() else "cpu"


class DataLoader:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.model = config.model
        self.task_prefix = config.task_prefix

        self.conversations, self.labels = self._load_conversations()
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        (
            self.train_conversations,
            self.test_conversations,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            self.conversations,
            self.encoded_labels,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=self.encoded_labels,
        )

        print(f"Loaded {len(self.conversations)} conversations")
        print(f"Classes: {self.label_encoder.classes_}")
        print(
            f"Train: {len(self.train_conversations)}, Test: {len(self.test_conversations)}"
        )

    def _load_conversations(self) -> Tuple[List[str], List[str]]:
        conversations = []
        labels = []

        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))

        for file_path in txt_files:
            filename = os.path.basename(file_path)

            if "_" in filename:
                parts = filename.replace(".txt", "").split("_")
                if len(parts) >= 2:
                    prefix = parts[0]
                    if prefix != self.task_prefix:
                        continue
                    value = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]

                    with open(file_path, "r", encoding="utf-8") as f:
                        conversation = f.read().strip()

                    conversations.append(conversation)
                    labels.append(value)

        return conversations, labels

    def extract_human_messages(self, conversation: str) -> str:
        return conversation

    def get_activations(self, texts: List[str], layer: int) -> torch.Tensor:
        activations = []

        for text in texts:
            tail = TAILS.get(self.task_prefix)
            cleaned = sanitize_conversation(text)
            safe_text = truncate_to_tail(cleaned, tail) if tail else cleaned
            tokens = self.model.tokenizer(
                safe_text, return_tensors="pt", truncation=True, max_length=1024
            )

            with torch.no_grad():
                with self.model.trace(tokens["input_ids"]):
                    layer_output = self.model.model.layers[layer].output
                    hidden_states = (
                        layer_output[0]
                        if isinstance(layer_output, tuple)
                        else layer_output
                    )
                    activation = hidden_states[:, -1, :].save()

            activations.append(activation.cpu())

        return torch.cat(activations, dim=0)

    def get_train_batch(
        self, layer: int, batch_size: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.config.batch_size

        indices = np.random.choice(
            len(self.train_conversations), size=batch_size, replace=False
        )
        batch_conversations = [self.train_conversations[i] for i in indices]
        batch_labels = torch.tensor(
            [self.train_labels[i] for i in indices], dtype=torch.long
        )

        activations = self.get_activations(batch_conversations, layer)

        return activations, batch_labels

    def get_test_data(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = self.get_activations(self.test_conversations, layer)
        labels = torch.tensor(self.test_labels, dtype=torch.long)
        return activations, labels


class LinearProbe(torch.nn.Module):
    def __init__(self, d_in: int, n_cls: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(d_in, n_cls, bias=bias)

    def forward(self, x):
        return self.W(x)


class LinearProbeTrainer:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.dataloader = DataLoader(config)

        self.d_model = config.model.config.hidden_size
        self.n_classes = len(self.dataloader.label_encoder.classes_)

        # Initialized per-layer in train_for_layer
        self.probe = None
        self.optimizer = None

        print(f"Probe input dim: {self.d_model}, output classes: {self.n_classes}")
        print(f"Classes: {self.dataloader.label_encoder.classes_}")

        # Where to save trained probes and metadata
        self.artifacts_dir = Path("artifacts") / self.config.task_prefix
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_artifacts(self, layer: int, metrics: Dict[str, float]):
        """Save the trained probe weights, classes, and metrics for this layer."""
        # Save probe weights
        weights_path = self.artifacts_dir / f"layer_{layer:02d}.pt"
        torch.save(self.probe.state_dict(), weights_path)

        # Save classes (label order) once per task
        classes_path = self.artifacts_dir / "classes.json"
        classes = list(self.dataloader.label_encoder.classes_)
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(classes, f)

        # Append metrics for this layer
        metrics_path = self.artifacts_dir / "metrics.jsonl"
        rec = {"layer": layer, **metrics}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def get_loss(self, layer: int) -> torch.Tensor:
        activations, labels = self.dataloader.get_train_batch(layer)
        activations, labels = activations.to(device), labels.to(device)

        logits = self.probe(activations)
        loss = F.cross_entropy(logits, labels)

        return loss

    def evaluate(self, layer: int) -> Dict[str, float]:
        self.probe.eval()

        with torch.no_grad():
            test_activations, test_labels = self.dataloader.get_test_data(layer)
            test_activations, test_labels = (
                test_activations.to(device),
                test_labels.to(device),
            )

            logits = self.probe(test_activations)
            test_loss = F.cross_entropy(logits, test_labels)

            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == test_labels).float().mean()

        self.probe.train()
        return {"test_loss": test_loss.item(), "test_accuracy": accuracy.item()}

    def train_for_layer(self, layer: int):
        self.probe = LinearProbe(self.d_model, self.n_classes).to(device)
        self.optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=self.config.learning_rate
        )

        print(
            f"Training probe for task={self.config.task_prefix}, layer={layer}, steps={self.config.steps}..."
        )

        # Initialize a dedicated W&B run per probe (task+layer)
        if self.config.use_wandb:
            cfg_dict = {k: v for k, v in vars(self.config).items() if k != "model"}
            cfg_dict["task"] = self.config.task_prefix
            cfg_dict["layer"] = layer
            run_name = f"probe-{self.config.task_prefix}-L{layer:02d}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                group=self.config.task_prefix,
                job_type="linear-probe",
                config=cfg_dict,
                reinit=True,
            )
            # Make sure charts use our logged "step" as x-axis
            try:
                wandb.define_metric("step")
                wandb.define_metric("*", step="step")
            except Exception:
                pass

        for step in tqdm(range(self.config.steps)):
            loss = self.get_loss(layer)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            metrics = {"train_loss": loss.item(), "step": step, "layer": layer}

            if step % 10 == 0 or step == self.config.steps - 1:
                eval_metrics = self.evaluate(layer)
                metrics.update(eval_metrics)

                print(
                    f"Layer {layer} Step {step}: Loss={loss.item():.4f}, Test Acc={eval_metrics['test_accuracy']:.4f}"
                )

            if self.config.use_wandb:
                # Log with explicit step so charts line up per run
                wandb.log(metrics, step=step)

        final_metrics = self.evaluate(layer)
        print(f"Layer {layer} Final - Test Loss: {final_metrics['test_loss']:.4f}")
        print(
            f"Layer {layer} Final - Test Accuracy: {final_metrics['test_accuracy']:.4f}"
        )

        # Persist artifacts for later use
        self.save_artifacts(layer, final_metrics)

        if self.config.use_wandb:
            wandb.log(
                {"final_test_accuracy": final_metrics["test_accuracy"], "layer": layer},
                step=self.config.steps - 1,
            )
            # Summarize key outcomes for quick comparison across runs
            wandb.summary["final_test_loss"] = final_metrics["test_loss"]
            wandb.summary["final_test_accuracy"] = final_metrics["test_accuracy"]
            wandb.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train linear probes across layers")
    parser.add_argument(
        "--task",
        choices=["socioeco", "religion", "location", "all"],
        default="all",
        help="Which task/tail to train (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Training steps per layer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    base_config = ProbeConfig(
        steps=args.steps,
        layer=31,
        use_wandb=not args.no_wandb,
        batch_size=args.batch_size,
        learning_rate=1e-3,
    )

    # Train probes across all layers for each task
    tasks = ["socioeco", "religion", "location"] if args.task == "all" else [args.task]
    n_layers = base_config.model.config.num_hidden_layers
    layers = list(range(n_layers))

    for task in tasks:
        print(f"\n=== Training task: {task} ===")
        cfg = ProbeConfig(**{**vars(base_config), "task_prefix": task})
        trainer = LinearProbeTrainer(cfg)

        for layer in layers:
            trainer.train_for_layer(layer)



if __name__ == "__main__":
    main()
