import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import glob
import numpy as np
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import wandb

cache_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf_cache"
)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir


def sanitize_conversation(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").replace("\r\n", "\n").split("\n")]
    kept = [
        ln
        for ln in lines
        if ln.startswith("### Human:") or ln.startswith("### Assistant:")
    ]
    return "\n".join(kept).strip() or text


def prefix_until_last_human(text: str) -> str:
    lines = (text or "").replace("\r\n", "\n").split("\n")
    role_lines = [
        ln.strip()
        for ln in lines
        if ln.strip().startswith("### Human:") or ln.strip().startswith("### Assistant:")
    ]
    last_h_idx = -1
    for i, ln in enumerate(role_lines):
        if ln.startswith("### Human:"):
            last_h_idx = i
    if last_h_idx == -1:
        return sanitize_conversation(text)
    return "\n".join(role_lines[: last_h_idx + 1]).strip()


@dataclass
class SteeringConfig:
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
    artifacts_root: str = "artifacts_control"  # separate from reading probes


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class SteeringData:
    def __init__(self, config: SteeringConfig):
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

        print(f"Loaded {len(self.conversations)} conversations (task={self.task_prefix})")
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

    def get_activations_at_last_user(self, texts: List[str], layer: int) -> torch.Tensor:
        activations = []

        for text in texts:
            cleaned = sanitize_conversation(text)
            prefix = prefix_until_last_human(cleaned)
            tokens = self.model.tokenizer(
                prefix, return_tensors="pt", truncation=True, max_length=1024
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

        activations = self.get_activations_at_last_user(batch_conversations, layer)
        return activations, batch_labels

    def get_test_data(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = self.get_activations_at_last_user(self.test_conversations, layer)
        labels = torch.tensor(self.test_labels, dtype=torch.long)
        return activations, labels


class LinearProbe(torch.nn.Module):
    def __init__(self, d_in: int, n_cls: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(d_in, n_cls, bias=bias)

    def forward(self, x):
        return self.W(x)


class SteeringProbeTrainer:
    def __init__(self, config: SteeringConfig):
        self.config = config
        self.data = SteeringData(config)

        self.d_model = config.model.config.hidden_size
        self.n_classes = len(self.data.label_encoder.classes_)

        self.probe = None
        self.optimizer = None

        print(f"Probe input dim: {self.d_model}, output classes: {self.n_classes}")
        print(f"Classes: {self.data.label_encoder.classes_}")

        # Save control probes in a distinct directory tree
        self.artifacts_dir = Path(self.config.artifacts_root) / self.config.task_prefix
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_artifacts(self, layer: int, metrics: Dict[str, float]):
        weights_path = self.artifacts_dir / f"layer_{layer:02d}.pt"
        torch.save(self.probe.state_dict(), weights_path)

        classes_path = self.artifacts_dir / "classes.json"
        classes = list(self.data.label_encoder.classes_)
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(classes, f)

        metrics_path = self.artifacts_dir / "metrics.jsonl"
        rec = {"layer": layer, **metrics}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def get_loss(self, layer: int) -> torch.Tensor:
        acts, labels = self.data.get_train_batch(layer)
        acts, labels = acts.to(device), labels.to(device)
        logits = self.probe(acts)
        loss = F.cross_entropy(logits, labels)
        return loss

    def evaluate(self, layer: int) -> Dict[str, float]:
        self.probe.eval()
        with torch.no_grad():
            acts, labels = self.data.get_test_data(layer)
            acts, labels = acts.to(device), labels.to(device)
            logits = self.probe(acts)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean()
        self.probe.train()
        return {"test_loss": loss.item(), "test_accuracy": acc.item()}

    def train_for_layer(self, layer: int):
        self.probe = LinearProbe(self.d_model, self.n_classes).to(device)
        self.optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=self.config.learning_rate
        )

        print(
            f"Training CONTROL probe for task={self.config.task_prefix}, layer={layer}, steps={self.config.steps}..."
        )

        if self.config.use_wandb:
            cfg_dict = {k: v for k, v in vars(self.config).items() if k != "model"}
            cfg_dict["task"] = self.config.task_prefix
            cfg_dict["layer"] = layer
            run_name = f"control-probe-{self.config.task_prefix}-L{layer:02d}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                group=f"control-{self.config.task_prefix}",
                job_type="control-linear-probe",
                config=cfg_dict,
                reinit=True,
            )
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
                wandb.log(metrics, step=step)

        final = self.evaluate(layer)
        print(f"Layer {layer} Final - Test Loss: {final['test_loss']:.4f}")
        print(f"Layer {layer} Final - Test Accuracy: {final['test_accuracy']:.4f}")

        self.save_artifacts(layer, final)

        if self.config.use_wandb:
            wandb.log({"final_test_accuracy": final["test_accuracy"], "layer": layer}, step=self.config.steps - 1)
            wandb.summary["final_test_loss"] = final["test_loss"]
            wandb.summary["final_test_accuracy"] = final["test_accuracy"]
            wandb.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train control (steering) linear probes across layers"
    )
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

    base_config = SteeringConfig(
        steps=args.steps,
        layer=31,
        use_wandb=not args.no_wandb,
        batch_size=args.batch_size,
        learning_rate=1e-3,
    )

    tasks = ["socioeco", "religion", "location"] if args.task == "all" else [args.task]
    n_layers = base_config.model.config.num_hidden_layers
    layers = list(range(n_layers))

    for task in tasks:
        print(f"\n=== Training CONTROL task: {task} ===")
        cfg = SteeringConfig(**{**vars(base_config), "task_prefix": task})
        trainer = SteeringProbeTrainer(cfg)

        for layer in layers:
            trainer.train_for_layer(layer)


if __name__ == "__main__":
    main()
