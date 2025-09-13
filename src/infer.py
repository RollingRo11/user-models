import os
import json
from pathlib import Path
from typing import List, Union, Dict

import torch
from nnsight import LanguageModel

from probe import (
    LinearProbe,
    TAILS,
    truncate_to_tail,
    sanitize_conversation,
)


class ProbeInference:
    def __init__(
        self,
        task: str,
        layer: int,
        artifacts_dir: Union[str, os.PathLike] = "artifacts",
        model_id: str = "meta-llama/Meta-Llama-3.1-8B",
        device: str = None,
    ):
        if task not in TAILS:
            raise ValueError(
                f"Unknown task '{task}'. Expected one of: {list(TAILS.keys())}"
            )

        self.task = task
        self.layer = layer
        self.tail = TAILS[task]
        self.artifacts_dir = Path(artifacts_dir) / task

        self.model = LanguageModel(model_id, device_map="auto")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        classes_path = self.artifacts_dir / "classes.json"
        if classes_path.exists():
            self.classes: List[str] = json.loads(
                classes_path.read_text(encoding="utf-8")
            )
        else:
            fallback = {
                "socioeco": ["low", "middle", "high"],
                "religion": ["christianity", "hinduism", "islam"],
                "location": ["europe", "north_america", "east_asia"],
            }
            if task not in fallback:
                raise FileNotFoundError(
                    f"Missing classes file at {classes_path} and no fallback for task '{task}'"
                )
            self.classes = sorted(fallback[task])

        d_model = self.model.config.hidden_size
        self.probe = LinearProbe(d_model, n_cls=len(self.classes))
        weights_path = self.artifacts_dir / f"layer_{self.layer:02d}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing probe weights for layer {self.layer}: {weights_path}"
            )
        self.probe.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.probe.to(self.device)
        self.probe.eval()

    def _activation_for(self, text: str) -> torch.Tensor:
        cleaned = sanitize_conversation(text)
        safe_text = truncate_to_tail(cleaned, self.tail)

        tokens = self.model.tokenizer(
            safe_text, return_tensors="pt", truncation=True, max_length=1024
        )
        with torch.no_grad():
            with self.model.trace(tokens["input_ids"]):
                layer_output = self.model.model.layers[self.layer].output
                hidden_states = (
                    layer_output[0] if isinstance(layer_output, tuple) else layer_output
                )
                activation = hidden_states[:, -1, :].save()

        return activation.cpu()  # keep on CPU; move later in batch

    def predict_proba(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        if isinstance(texts, str):
            texts = [texts]

        activations = [self._activation_for(t) for t in texts]
        X = torch.cat(activations, dim=0).to(self.device)

        with torch.no_grad():
            logits = self.probe(X)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        results: List[Dict[str, float]] = []
        for row in probs:
            results.append({cls: float(p) for cls, p in zip(self.classes, row)})
        return results


__all__ = ["ProbeInference"]
