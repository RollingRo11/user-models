"""
Generate Plotly HTML visualizations for linear probes (reading and control) and
religion-focused examples, saved under ./visualizations.

Edit the CONFIG section to change prompts, tasks, artifacts, and layer choices.
No argparse to keep it simple and easy to tweak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.io as pio
import plotly.graph_objects as go

# Import plotting helpers from the repo (used by the notebook as well)
from vis import (
    scan_wandb_runs_control,
    scan_wandb_runs_reading,
    plot_loss_lines,
    plot_accuracy_line,
    plot_control_loss_lines,
    plot_control_accuracy_line,
    plot_prompt_probe_heatmap,
    plot_prompt_last_layer_dual_heatmap,
    plot_probe_confusion,
)


# ============ CONFIG (easy to edit) ============

# Output directory for all figures
OUTPUT_DIR = Path("visualizations")

# Which task to focus on (religion | socioeco | location)
TASK = "religion"

# Artifact roots for reading vs control
READING_ROOT = Path("artifacts")
CONTROL_ROOT = Path("artifacts_control")

# Optional: pin a specific layer index for last-layer comparisons (None = use last available)
PINNED_LAYER: Optional[int] = None

# Model selection used to extract activations for prompt heatmaps
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# Device map: None lets the helper auto-choose; set to 'cpu' to force CPU
DEVICE_MAP: Optional[str] = None

# Prompts used in the notebook that are religion-oriented
PROMPTS: Dict[str, str] = {
    # Masjid example — Muslim context
    "masjid": (
        "\n"
        "### Human: OMG! I am so excited to go to the Masjid today!\n\n"
        "### Assistant: I think the user's religion is"
    ),
    # Gayatri example — Hindu context (same as notebook's `test_things` end tail)
    "gayatri": (
        "### Human: I always find it calming to start the day by quietly reciting the Gayatri at sunrise,\n"
        "even if it’s just under my breath for a few moments.\n\n"
        "### Assistant: That’s a beautiful way to center yourself at dawn The Gayatri is meant exactly for that—quiet reflection,\n"
        "renewal, and connection as the sun rises. Even a few breaths with it under your breath can set a peaceful tone for the whole day.\n\n"
        "### Human: “Yes, exactly. Sometimes I also pause at dusk with a small prayer—it feels like closing the circle of the day.”\n\n"
        "### Assistant: I think the user's religion is"
    ),
}

"""
Religion-specific variations demonstrating control to a different class than the
prompt implies. Keys encode base religion and control target, e.g.,
"hindu_gayatri_to_islam" means the prompt suggests Hinduism but we examine
control effects targeting Islam.
Each entry has:
  - text: prompt string
  - base: implied class in the prompt
  - target: control target class (different from base)
"""
RELIGION_VARIATIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    "religion": {
        # Prompts implying Islam, controlled toward Christianity/Hinduism
        "islam_masjid_to_christianity": {
            "text": (
                "### Human: I'm heading to the masjid for Jummah prayer—can't be late!\n\n"
                "### Assistant: I think the user's religion is"
            ),
            "base": "islam",
            "target": "christianity",
        },
        "islam_ramadan_to_hinduism": {
            "text": (
                "### Human: Fasting has been tough this Ramadan, but the iftar gatherings make it special.\n\n"
                "### Assistant: I think the user's religion is"
            ),
            "base": "islam",
            "target": "hinduism",
        },

        # Prompts implying Christianity, controlled toward Islam/Hinduism
        "christian_sunday_mass_to_islam": {
            "text": (
                "### Human: We’re attending Sunday Mass and then choir practice.\n\n"
                "### Assistant: I think the user's religion is"
            ),
            "base": "christianity",
            "target": "islam",
        },
        "christian_bible_study_to_hinduism": {
            "text": (
                "### Human: Our Bible study group is discussing the Sermon on the Mount tonight.\n\n"
                "### Assistant: I think the user's religion is"
            ),
            "base": "christianity",
            "target": "hinduism",
        },

        # Prompts implying Hinduism, controlled toward Islam/Christianity
        "hindu_gayatri_to_islam": {
            "text": (
                "### Human: I recite the Gayatri mantra at sunrise—it keeps me centered.\n\n"
                "### Assistant: I think the user's religion is"
            ),
            "base": "hinduism",
            "target": "islam",
        },
        "hindu_diwali_puja_to_christianity": {
            "text": (
                "### Human: We’re preparing for Lakshmi puja tonight for Diwali.\n\n"
                "### Assistant: I think the user's religion is"
            ),
            "base": "hinduism",
            "target": "christianity",
        },
    }
}

# Page settings for write_html
PLOTLY_WRITE_KW = dict(include_plotlyjs="cdn", full_html=True)


# ============ Helpers ============

# Collect mapping of output file -> prompt and optional metadata (base/target)
MANIFEST: List[Dict[str, Optional[str]]] = []

def ensure_outdir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_fig(fig, name: str, prompt_text: Optional[str] = None, *, meta: Optional[Dict[str, str]] = None) -> None:
    out = ensure_outdir() / f"{name}.html"
    fig.write_html(str(out), **PLOTLY_WRITE_KW)
    print(f"[saved] {out}")
    rec = {"file": str(out), "prompt": prompt_text}
    if meta:
        rec.update(meta)
    MANIFEST.append(rec)


def last_available_layer(artifacts_root: Path, task: str) -> Optional[int]:
    task_dir = artifacts_root / task
    if not task_dir.exists():
        return None
    layers: List[int] = []
    for p in task_dir.glob("layer_*.pt"):
        try:
            num = int(p.stem.split("_")[1])
            layers.append(num)
        except Exception:
            continue
    return max(layers) if layers else None


# ============ Generators ============

def generate_wandb_overview() -> None:
    """Loss and accuracy lines for reading vs control probes (all tasks with runs)."""
    # Reading probe runs
    runs_read = scan_wandb_runs_reading(Path("wandb"))
    fig = plot_loss_lines(runs_read, dropdown_position="top-right", legend_bottom=False)
    save_fig(fig, "reading_loss_lines")
    fig = plot_accuracy_line(runs_read, dropdown_position="top-right")
    save_fig(fig, "reading_accuracy_line")

    # Control probe runs
    runs_ctrl = scan_wandb_runs_control(Path("wandb"))
    fig = plot_control_loss_lines(runs_ctrl, dropdown_position="top-right", legend_bottom=False)
    save_fig(fig, "control_loss_lines")
    fig = plot_control_accuracy_line(runs_ctrl, dropdown_position="top-right")
    save_fig(fig, "control_accuracy_line")


def generate_prompt_heatmaps(task: str, prompts: Dict[str, str]) -> None:
    """For each prompt, save:
    - Reading heatmap (layers x classes)
    - Control heatmap (layers x classes)
    - Dual last-layer heatmap (rows: reading vs control)
    """
    for key, prompt in prompts.items():
        # Reading across all available layers
        fig = plot_prompt_probe_heatmap(
            task,
            prompt,
            artifacts_root=str(READING_ROOT),
            model_id=MODEL_ID,
            device_map=DEVICE_MAP,
        )
        save_fig(fig, f"{task}_{key}_reading_layers_heatmap", prompt)

        # Control across all available layers
        fig = plot_prompt_probe_heatmap(
            task,
            prompt,
            artifacts_root=str(CONTROL_ROOT),
            model_id=MODEL_ID,
            device_map=DEVICE_MAP,
        )
        save_fig(fig, f"{task}_{key}_control_layers_heatmap", prompt)

        # Dual last-layer comparison (reading vs control)
        fig = plot_prompt_last_layer_dual_heatmap(
            task,
            prompt,
            reading_root=str(READING_ROOT),
            control_root=str(CONTROL_ROOT),
            model_id=MODEL_ID,
            device_map=DEVICE_MAP,
            layer=PINNED_LAYER,
        )
        save_fig(fig, f"{task}_{key}_last_layer_dual_heatmap", prompt)


def generate_confusions(task: str) -> None:
    """Generate 3x3 confusion matrix heatmaps for reading and control probes.

    Uses PINNED_LAYER if set, otherwise the last available layer in each artifacts root.
    Forces CPU for stability in batch runs.
    """
    # Choose layers
    read_L = PINNED_LAYER if PINNED_LAYER is not None else last_available_layer(READING_ROOT, task)
    ctrl_L = PINNED_LAYER if PINNED_LAYER is not None else last_available_layer(CONTROL_ROOT, task)
    if read_L is None:
        raise ValueError(f"No reading probe layers found for task '{task}' in {READING_ROOT}")
    if ctrl_L is None:
        raise ValueError(f"No control probe layers found for task '{task}' in {CONTROL_ROOT}")

    # Reading confusion
    fig = plot_probe_confusion(
        task,
        layer=int(read_L),
        artifacts_root=str(READING_ROOT),
        data_dir="data",
        model_id=MODEL_ID,
        device="cpu",
        device_map="cpu",
    )
    save_fig(fig, f"{task}_confusion_reading_L{int(read_L):02d}")

    # Control confusion
    fig = plot_probe_confusion(
        task,
        layer=int(ctrl_L),
        artifacts_root=str(CONTROL_ROOT),
        data_dir="data",
        model_id=MODEL_ID,
        device="cpu",
        device_map="cpu",
    )
    save_fig(fig, f"{task}_confusion_control_L{int(ctrl_L):02d}")


# -------- Extra (creative) visualizations for control prompts --------

def _compute_layer_probs(
    task: str,
    prompt: str,
    artifacts_root: Path,
    model_id: str,
    device_map: Optional[str],
) -> Tuple[List[int], List[str], List[List[float]]]:
    """Return (layers, classes, probs[layer_idx][class_idx]) for a single prompt.

    Mirrors logic used by vis.plot_prompt_probe_heatmap but returns raw data to build
    custom figures like deltas or trajectories.
    """
    from nnsight import LanguageModel
    import torch
    from probe import LinearProbe, TAILS, truncate_to_tail, sanitize_conversation
    from vis import _available_layers as available_layers, _load_classes as load_classes

    task_dir = artifacts_root / task
    classes = load_classes(task, task_dir)
    layers = available_layers(task_dir)
    if not layers:
        raise ValueError(f"No probe layers found in {task_dir}")

    # Model
    chosen_device_map = device_map if device_map is not None else "auto"
    model = LanguageModel(model_id, device_map=chosen_device_map)

    # Prepare text
    tail = TAILS.get(task)
    cleaned = sanitize_conversation(prompt)
    safe_text = truncate_to_tail(cleaned, tail) if tail else cleaned
    tokens = model.tokenizer(safe_text, return_tensors="pt", truncation=True, max_length=512)

    # Build all probes
    d_model = model.config.hidden_size
    n_cls = len(classes)
    # Evaluate probes on CPU to avoid GPU context issues in headless/batch runs
    device = "cpu"

    probes = {}
    for L in layers:
        lp = LinearProbe(d_model, n_cls)
        weights = torch.load(task_dir / f"layer_{L:02d}.pt", map_location=device)
        lp.load_state_dict(weights)
        lp.to(device).eval()
        probes[L] = lp

    # Collect per-layer probabilities
    z: List[List[float]] = []
    with torch.no_grad():
        with model.trace(tokens["input_ids"]):
            for L in layers:
                layer_output = model.model.layers[L].output
                hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                act = hs[:, -1, :].save().to(device)
                logits = probes[L](act)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
                z.append([float(probs[i]) for i in range(n_cls)])

    return layers, classes, z


def generate_control_creatives(task: str, variations: Dict[str, Dict[str, str]]) -> None:
    """Creative control-specific visuals per prompt:
    - Delta heatmap: (control - reading) probability per layer×class.
    - Target class trajectory: p(target) over layers (reading vs control).
    - Target margin over next-best: (p(target) - max(other)) over layers (reading vs control).
    """
    read_layers = last_available_layer(READING_ROOT, task)
    ctrl_layers = last_available_layer(CONTROL_ROOT, task)
    # Not strictly required beyond existence checks by loaders
    _ = read_layers, ctrl_layers

    for key, spec in variations.items():
        prompt = spec["text"]
        base = spec.get("base")
        target = spec.get("target") or spec.get("class")

        # Extract matrices for reading and control
        r_layers, classes, r_z = _compute_layer_probs(task, prompt, READING_ROOT, MODEL_ID, DEVICE_MAP)
        c_layers, c_classes, c_z = _compute_layer_probs(task, prompt, CONTROL_ROOT, MODEL_ID, DEVICE_MAP)

        # Align on common layers (assumes same indices/order OK when trained identically)
        if r_layers != c_layers:
            # Intersect while preserving order
            common = [L for L in r_layers if L in set(c_layers)]
            # Re-index z accordingly
            idx_map_r = [r_layers.index(L) for L in common]
            idx_map_c = [c_layers.index(L) for L in common]
            r_layers = common
            c_layers = common
            r_z = [r_z[i] for i in idx_map_r]
            c_z = [c_z[i] for i in idx_map_c]

        # 1) Delta heatmap (control - reading)
        import numpy as np
        rz = np.array(r_z)
        cz = np.array(c_z)
        dz = (cz - rz).tolist()
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=dz,
                    x=classes,
                    y=r_layers,
                    colorscale="RdBu",
                    zmid=0.0,
                    zmin=-1.0,
                    zmax=1.0,
                    colorbar=dict(title="Δ prob (control - reading)", thickness=12, len=0.9),
                    hovertemplate="layer=%{y}<br>class=%{x}<br>Δ=%{z}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=f"{task} — {key}: control minus reading probabilities over layers",
            xaxis_title="Class",
            yaxis_title="Layer",
            xaxis=dict(tickangle=45, automargin=True),
            margin=dict(l=60, r=60, t=80, b=80),
            width=700,
            height=max(420, 26 * len(r_layers) + 120),
        )
        save_fig(fig, f"{task}_{key}_delta_layers_heatmap", prompt, meta={"base": base, "target": target})

        # 2) Target trajectory and margin lines
        if target in classes:
            ti = classes.index(target)
            # p(target) per layer
            r_pt = [row[ti] for row in r_z]
            c_pt = [row[ti] for row in c_z]
            # margin per layer: p(target) - max(other)
            def margins(z_mat: List[List[float]]) -> List[float]:
                m: List[float] = []
                for row in z_mat:
                    others = [row[j] for j in range(len(row)) if j != ti]
                    m.append(row[ti] - max(others) if others else row[ti])
                return m
            r_m = margins(r_z)
            c_m = margins(c_z)

            # p(target) lines
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r_layers, y=r_pt, mode="lines+markers", name="reading"))
            fig.add_trace(go.Scatter(x=r_layers, y=c_pt, mode="lines+markers", name="control"))
            fig.update_layout(
                title=f"{task} — {key}: p({target}) over layers (reading vs control)",
                xaxis_title="Layer",
                yaxis_title=f"p({target})",
                yaxis=dict(range=[0, 1], rangemode="tozero"),
                margin=dict(l=60, r=40, t=70, b=60),
                width=720,
                height=420,
            )
            save_fig(fig, f"{task}_{key}_target_probability_lines", prompt, meta={"base": base, "target": target})

            # margin lines
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r_layers, y=r_m, mode="lines+markers", name="reading"))
            fig.add_trace(go.Scatter(x=r_layers, y=c_m, mode="lines+markers", name="control"))
            fig.update_layout(
                title=f"{task} — {key}: margin vs next-best for {target}",
                xaxis_title="Layer",
                yaxis_title=f"p({target}) - max(other)",
                yaxis=dict(rangemode="tozero"),
                margin=dict(l=60, r=40, t=70, b=60),
                width=720,
                height=420,
            )
            save_fig(fig, f"{task}_{key}_target_margin_lines", prompt, meta={"base": base, "target": target})
        else:
            print(f"[warn] target class '{target}' not found in classes {classes}")


def main() -> None:
    # No renderer needed for write_html; avoid setting one to prevent errors in headless envs
    ensure_outdir()

    # 1) Overview lines from W&B (works if you have offline runs in ./wandb)
    generate_wandb_overview()

    # 2) Religion-focused prompt visualizations
    generate_prompt_heatmaps(TASK, PROMPTS)

    # 3) Control-prompt (religion variations) creative visuals
    if TASK in RELIGION_VARIATIONS:
        generate_control_creatives(TASK, RELIGION_VARIATIONS[TASK])

    # 4) Confusion matrices (3x3) for reading and control
    generate_confusions(TASK)

    # 5) Write a simple manifest of outputs and prompt strings used
    manifest_path = ensure_outdir() / "prompts_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# Generated visualizations and associated prompt strings\n")
        f.write("# Lines with no prompt were not based on a single input string.\n")
        f.write("# For control creatives, base=prompt-implied class, target=control target class.\n\n")
        for rec in MANIFEST:
            f.write(f"file: {rec.get('file')}\n")
            text = rec.get("prompt")
            if text:
                f.write("prompt:\n")
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
            else:
                f.write("prompt: (none)\n")
            base = rec.get("base")
            target = rec.get("target")
            if base or target:
                f.write(f"base: {base or '(n/a)'}\n")
                f.write(f"target: {target or '(n/a)'}\n")
            f.write("---\n")
    print(f"[saved] {manifest_path}")

    print("Done. Explore ./visualizations for the HTML outputs.")


if __name__ == "__main__":
    main()
