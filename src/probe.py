import os
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

cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.hf_cache')
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir


@dataclass
class ProbeConfig:
    steps: int = 50
    layer: int = 31
    use_wandb: bool = True
    wandb_project: str = "user-models"
    model: LanguageModel = field(default_factory=lambda: LanguageModel(
        "meta-llama/Meta-Llama-3.1-8B", device_map='auto'))
    data_dir: str = "data"
    batch_size: int = 32
    learning_rate: float = 1e-3
    test_size: float = 0.2
    random_seed: int = 42


device = "cuda" if torch.cuda.is_available() else "cpu"


class DataLoader:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.model = config.model

        self.conversations, self.labels = self._load_conversations()
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.train_conversations, self.test_conversations, \
            self.train_labels, self.test_labels = train_test_split(
                self.conversations, self.encoded_labels,
                test_size=config.test_size,
                random_state=config.random_seed,
                stratify=self.encoded_labels
            )

        print(f"Loaded {len(self.conversations)} conversations")
        print(f"Classes: {self.label_encoder.classes_}")
        print(
            f"Train: {len(self.train_conversations)}, Test: {len(self.test_conversations)}")

    def _load_conversations(self) -> Tuple[List[str], List[str]]:
        conversations = []
        labels = []

        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))

        for file_path in txt_files:
            filename = os.path.basename(file_path)

            if "_" in filename:
                parts = filename.replace(".txt", "").split("_")
                if len(parts) >= 2:
                    category = "_".join(parts[:-1])

                    with open(file_path, "r", encoding="utf-8") as f:
                        conversation = f.read().strip()

                    conversations.append(conversation)
                    labels.append(category)

        return conversations, labels

    def extract_human_messages(self, conversation: str) -> str:
        lines = conversation.split("\n")
        human_messages = []

        for line in lines:
            if line.strip().startswith("### Human:"):
                message = line.replace("### Human:", "").strip()
                human_messages.append(message)

        return " ".join(human_messages)

    def get_activations(self, texts: List[str], layer: int) -> torch.Tensor:
        activations = []

        for text in texts:
            human_text = self.extract_human_messages(text)

            tokens = self.model.tokenizer(human_text, return_tensors="pt",
                                          truncation=True, max_length=512)

            with torch.no_grad():
                with self.model.trace(tokens["input_ids"]):
                    layer_output = self.model.model.layers[layer].output
                    hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                    activation = hidden_states.mean(dim=1).save()

            activations.append(activation.cpu())

        return torch.cat(activations, dim=0)

    def get_train_batch(self, batch_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.config.batch_size

        indices = np.random.choice(
            len(self.train_conversations), size=batch_size, replace=False)
        batch_conversations = [self.train_conversations[i] for i in indices]
        batch_labels = torch.tensor([self.train_labels[i]
                                    for i in indices], dtype=torch.long)

        activations = self.get_activations(
            batch_conversations, self.config.layer)

        return activations, batch_labels

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = self.get_activations(
            self.test_conversations, self.config.layer)
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

        d_model = config.model.config.hidden_size
        n_classes = len(self.dataloader.label_encoder.classes_)

        self.probe = LinearProbe(d_model, n_classes).to(device)
        self.optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=config.learning_rate)

        print(f"Probe input dim: {d_model}, output classes: {n_classes}")
        print(f"Classes: {self.dataloader.label_encoder.classes_}")

    def get_loss(self) -> torch.Tensor:
        activations, labels = self.dataloader.get_train_batch()
        activations, labels = activations.to(device), labels.to(device)

        logits = self.probe(activations)
        loss = F.cross_entropy(logits, labels)

        return loss

    def evaluate(self) -> Dict[str, float]:
        self.probe.eval()

        with torch.no_grad():
            test_activations, test_labels = self.dataloader.get_test_data()
            test_activations, test_labels = test_activations.to(
                device), test_labels.to(device)

            logits = self.probe(test_activations)
            test_loss = F.cross_entropy(logits, test_labels)

            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == test_labels).float().mean()

        self.probe.train()
        return {"test_loss": test_loss.item(), "test_accuracy": accuracy.item()}

    def train(self):
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, config=self.config)

        print(f"Training probe for {self.config.steps} steps...")

        for step in tqdm(range(self.config.steps)):
            loss = self.get_loss()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            metrics = {"train_loss": loss.item(), "step": step}

            if step % 10 == 0 or step == self.config.steps - 1:
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)

                print(f"Step {step}: Loss={loss.item():.4f}, "
                      f"Test Acc={eval_metrics['test_accuracy']}")

            if self.config.use_wandb:
                wandb.log(metrics)

        final_metrics = self.evaluate()
        print(f"Test Loss: {final_metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {final_metrics['test_accuracy']:.4f}")

        if self.config.use_wandb:
            wandb.log({"final_test_accuracy": final_metrics['test_accuracy']})
            wandb.finish()


def main():
    config = ProbeConfig(
        steps=100,
        layer=31,
        use_wandb=True,
        batch_size=16,
        learning_rate=1e-3
    )

    trainer = LinearProbeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
