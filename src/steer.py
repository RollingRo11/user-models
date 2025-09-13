import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional

import torch
from nnsight import LanguageModel


def _parse_layers(spec: str) -> List[int]:
    """Parse layer spec like '20-29' or '15,18,22-24' into a raw list.

    Clamping to model depth happens later after model is loaded.
    """
    layers: List[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            start = int(a)
            end = int(b)
            layers.extend(list(range(start, end + 1)))
        else:
            layers.append(int(p))
    return layers


def _ensure_chat_prompt(text: str) -> str:
    """If `text` doesn't include chat markers, wrap it as a single turn.

    Produces:
    ### Human: {text}\n
    ### Assistant:
    """
    if "### Human:" in text or "### Assistant:" in text:
        return text
    # Normalize whitespace and wrap
    user = text.strip()
    return f"### Human: {user}\n### Assistant:"


@dataclass
class SteeringSpec:
    task: str
    target_class: str
    strength: float = 8.0
    layers: Optional[List[int]] = None  # default to middle layers 20-29
    artifacts_root: str = "artifacts_control"
    model_id: str = "meta-llama/Meta-Llama-3.1-8B"


class ControlProbeSteerer:
    def __init__(self, spec: SteeringSpec):
        self.spec = spec
        self.model = LanguageModel(
            spec.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = self.model.tokenizer
        self.hf = self.model._model  # underlying HF model for generation
        self.n_layers = self.hf.config.num_hidden_layers

        # Default layers to 20-29, then clamp to model depth and uniq
        raw_layers = (
            spec.layers if spec.layers is not None else list(range(20, 30))
        )
        seen = set()
        self.layers: List[int] = []
        for l in raw_layers:
            if 0 <= l < self.n_layers and l not in seen:
                self.layers.append(l)
                seen.add(l)

        # Load class list and per-layer probe weights
        self.artifacts_dir = Path(spec.artifacts_root) / spec.task
        classes_path = self.artifacts_dir / "classes.json"
        if not classes_path.exists():
            raise FileNotFoundError(f"Missing classes file: {classes_path}")
        self.classes: List[str] = json.loads(classes_path.read_text(encoding="utf-8"))
        if spec.target_class not in self.classes:
            raise ValueError(
                f"Unknown target_class '{spec.target_class}'. Available: {self.classes}"
            )
        self.target_idx = self.classes.index(spec.target_class)

        # Load direction vectors per layer from the probe's weight for target class
        self.directions: Dict[int, torch.Tensor] = {}
        for layer in self.layers:
            w = self._load_probe_weight_for_layer(layer)
            vec = w[self.target_idx].detach().float()  # shape [d_model]
            # Normalize to unit vector to control magnitude by strength only
            vec = vec / (vec.norm(p=2) + 1e-12)
            self.directions[layer] = vec

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _load_probe_weight_for_layer(self, layer: int) -> torch.Tensor:
        """Return W.weight tensor of shape [n_cls, d_model] for given layer.

        We read the state_dict and extract 'W.weight' without instantiating the
        probe module. This avoids needing the exact class definition.
        """
        weights_path = self.artifacts_dir / f"layer_{layer:02d}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing control probe weights: {weights_path}")
        sd = torch.load(weights_path, map_location="cpu")
        if "W.weight" not in sd:
            # Fallback: some save formats might nest keys differently
            # Try to find the first weight-like key
            for k in list(sd.keys()):
                if k.endswith("weight") and sd[k].dim() == 2:
                    return sd[k].clone().detach()
            raise KeyError(
                f"State dict at {weights_path} missing 'W.weight' (found keys: {list(sd.keys())[:5]}...)"
            )
        return sd["W.weight"].clone().detach()

    def _register_hooks(self):
        # Remove any existing hooks first (idempotent)
        self.remove_hooks()

        for layer in self.layers:
            module = self.hf.model.layers[layer]
            vec = self.directions[layer]

            def make_hook(v: torch.Tensor):
                def _hook(mod, inputs, output):
                    # output can be Tensor or tuple; adjust last-token hidden state
                    if isinstance(output, tuple):
                        hs = output[0]
                        rest = output[1:]
                    else:
                        hs = output
                        rest = None

                    # Ensure dtype/device alignment
                    v_dev = v.to(hs.device, dtype=hs.dtype)

                    # Add steering vector to last token only
                    adjusted = hs
                    # If seq len is 0 (should not happen), skip
                    if adjusted.size(1) > 0:
                        adjusted = adjusted.clone()
                        adjusted[:, -1, :] = adjusted[:, -1, :] + self.spec.strength * v_dev

                    if rest is None:
                        return adjusted
                    else:
                        return (adjusted,) + rest

                return _hook

            self._hooks.append(module.register_forward_hook(make_hook(vec)))

    def remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask")
        prompt_len = input_ids.shape[1]

        # Install hooks and generate under torch.no_grad()
        self._register_hooks()
        try:
            with torch.no_grad():
                gen_ids = self.hf.generate(
                    input_ids=input_ids.to(self.hf.device),
                    attention_mask=attention_mask.to(self.hf.device) if attention_mask is not None else None,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            # Always clean up hooks to avoid cross-run leakage
            self.remove_hooks()

        # Decode only the newly generated continuation
        continuation_ids = gen_ids[0][prompt_len:]
        text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return text.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply control-probe steering during generation")
    parser.add_argument("--task", required=True, choices=["socioeco", "religion", "location"], help="Attribute/task of the control probe")
    parser.add_argument("--target-class", required=True, help="Target class label to steer toward (must match classes.json)")
    parser.add_argument("--strength", type=float, default=8.0, help="Steering strength (N); paper used N=8")
    parser.add_argument("--layers", type=str, default="20-29", help="Layers to steer, e.g. '20-29' or '18,20,22-24'")
    parser.add_argument("--artifacts-root", type=str, default="artifacts_control", help="Directory containing trained control probes")
    parser.add_argument("--text", type=str, default=None, help="Prompt text. If omitted, uses --file or interactive input")
    parser.add_argument("--file", type=str, default=None, help="Path to a text file containing the prompt")
    parser.add_argument("--interactive", action="store_true", help="Read a single-line prompt from stdin and wrap as chat")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # Load prompt
    if args.text is not None:
        user_text = args.text
    elif args.file is not None:
        user_text = Path(args.file).read_text(encoding="utf-8")
    else:
        # Interactive single prompt (no markers needed)
        if not args.interactive:
            print("No --text or --file provided; entering one-shot interactive mode.\n")
        try:
            user_text = input("Enter your message: ").strip()
        except EOFError:
            raise SystemExit("No input provided.")

    # Ensure chat markers so generation continues as Assistant
    prompt = _ensure_chat_prompt(user_text)

    # Build steering spec
    raw_layers = _parse_layers(args.layers)

    spec = SteeringSpec(
        task=args.task,
        target_class=args.target_class,
        strength=args.strength,
        layers=raw_layers,
        artifacts_root=args.artifacts_root,
    )

    steerer = ControlProbeSteerer(spec)
    out = steerer.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )
    print(out)


if __name__ == "__main__":
    main()
