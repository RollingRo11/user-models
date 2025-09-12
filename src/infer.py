import os
import json
from pathlib import Path
from typing import List, Union, Dict

import torch
from nnsight import LanguageModel

# Reuse probing utilities and probe definition
from .probe import (
    LinearProbe,
    TAILS,
    truncate_to_tail,
    sanitize_conversation,
)


class ProbeInference:
    """Load a saved linear probe and run probability predictions.

    Usage:
        runner = ProbeInference(task="religion", layer=7)
        probs = runner.predict_proba("""
            ### Human: ...
            ### Assistant: I think the user's religion is
        """)
        # probs -> list of {class: probability}
    """

    def __init__(
        self,
        task: str,
        layer: int,
        artifacts_dir: Union[str, os.PathLike] = "artifacts",
        model_id: str = "meta-llama/Meta-Llama-3.1-8B",
        device: str = None,
    ):
        if task not in TAILS:
            raise ValueError(f"Unknown task '{task}'. Expected one of: {list(TAILS.keys())}")

        self.task = task
        self.layer = layer
        self.tail = TAILS[task]
        self.artifacts_dir = Path(artifacts_dir) / task

        # Model and device
        self.model = LanguageModel(model_id, device_map="auto")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load label classes
        classes_path = self.artifacts_dir / "classes.json"
        if not classes_path.exists():
            raise FileNotFoundError(f"Missing classes file: {classes_path}")
        self.classes: List[str] = json.loads(classes_path.read_text(encoding="utf-8"))

        # Build probe and load weights
        d_model = self.model.config.hidden_size
        self.probe = LinearProbe(d_model, n_cls=len(self.classes))
        weights_path = self.artifacts_dir / f"layer_{self.layer:02d}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing probe weights for layer {self.layer}: {weights_path}")
        self.probe.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.probe.to(self.device)
        self.probe.eval()

    def _activation_for(self, text: str) -> torch.Tensor:
        """Compute last-token activation at the configured layer for a single input."""
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
        """Return class probability dicts for one or more inputs.

        - Inputs should be conversations ending with the task-specific tail.
        - Sanitization and truncation are applied automatically.
        """
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

