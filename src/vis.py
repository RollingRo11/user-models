from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go



def find_wandb_dir(start: Optional[Path] = None) -> Optional[Path]:
    """Search upwards from `start` (or CWD) for a `wandb/` directory."""
    base = Path.cwd() if start is None else Path(start)
    for root in [base] + list(base.parents):
        cand = root / "wandb"
        if cand.exists() and cand.is_dir():
            return cand
    return None


def find_artifacts_dir(task: str, start: Optional[Path] = None) -> Optional[Path]:
    """Search upwards for `artifacts/<task>`; also try module-relative fallback."""
    base = Path.cwd() if start is None else Path(start)
    task = (task or "").strip()
    for root in [base] + list(base.parents):
        cand = root / "artifacts" / task
        if cand.exists() and cand.is_dir():
            return cand
    # Try module-relative: repo_root ≈ vis.py/../..
    try:
        mod_root = Path(__file__).resolve().parent
        cand = mod_root.parent / "artifacts" / task
        if cand.exists() and cand.is_dir():
            return cand
    except Exception:
        pass
    return None


def find_data_dir(task: str, start: Optional[Path] = None) -> Optional[Path]:
    """Search upwards for a `data/` directory containing `<task>_*.txt` files.

    Returns the directory path if found, else None.
    """
    base = Path.cwd() if start is None else Path(start)
    task = (task or "").strip()
    for root in [base] + list(base.parents):
        cand = root / "data"
        if cand.exists() and cand.is_dir():
            if any(cand.glob(f"{task}_*.txt")):
                return cand
    # Try module-relative
    try:
        mod_root = Path(__file__).resolve().parent
        cand = mod_root.parent / "data"
        if cand.exists() and cand.is_dir() and any(cand.glob(f"{task}_*.txt")):
            return cand
    except Exception:
        pass
    return None


def _parse_task_from_config(config_path: Path) -> Optional[str]:
    """Extract `task: value: <segment>` from a W&B config.yaml file.

    Falls back to None if not found.
    """
    try:
        lines = config_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return None

    for i, line in enumerate(lines):
        if line.strip().startswith("task:"):
            for j in range(i + 1, min(i + 6, len(lines))):
                val = lines[j].strip()
                if val.startswith("value:"):
                    raw = val.split(":", 1)[1].strip()
                    if (raw.startswith('"') and raw.endswith('"')) or (
                        raw.startswith("'") and raw.endswith("'")
                    ):
                        raw = raw[1:-1]
                    return raw
            break
    return None


def _parse_metadata_program(files_dir: Path) -> Optional[str]:
    """Return the script/program path recorded by W&B (from wandb-metadata.json)."""
    meta_path = files_dir / "wandb-metadata.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
        return meta.get("program")
    except Exception:
        return None


def scan_wandb_runs(
    wandb_dir: Optional[Path] = None,
    *,
    filter_script_suffix: Optional[str] = None,
) -> pd.DataFrame:
    """Scan offline W&B runs and collect per-layer metrics.

    Returns a DataFrame with columns:
    - segment, layer, train_loss, test_loss, test_accuracy, _timestamp, run_path
    Deduplicates by (segment, layer) keeping the latest by _timestamp.
    """
    if wandb_dir is None:
        wandb_dir = find_wandb_dir()
    if wandb_dir is None or not Path(wandb_dir).exists():
        return pd.DataFrame()

    records: List[Dict] = []
    for run_dir in Path(wandb_dir).iterdir():
        if not run_dir.is_dir():
            continue
        files_dir = run_dir / "files"
        summary_path = files_dir / "wandb-summary.json"
        config_path = files_dir / "config.yaml"
        if not summary_path.exists():
            continue

        # Optional filter by the launching script (e.g., 'src/probe.py' vs 'src/steering_probe.py')
        if filter_script_suffix is not None:
            prog = _parse_metadata_program(files_dir) or ""
            if not str(prog).endswith(filter_script_suffix):
                continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        layer = summary.get("layer")
        if layer is None:
            continue

        segment = _parse_task_from_config(config_path) or "unknown"
        rec = {
            "segment": segment,
            "layer": int(layer),
            "train_loss": summary.get("train_loss"),
            "test_loss": summary.get("test_loss") or summary.get("final_test_loss"),
            "test_accuracy": summary.get("test_accuracy")
            or summary.get("final_test_accuracy"),
            "_timestamp": summary.get("_timestamp", 0),
            "run_path": str(run_dir),
        }
        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # Cast numeric and dedupe by (segment, layer)
    for c in ("train_loss", "test_loss", "test_accuracy"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce").astype("Int64")

    df = (
        df.sort_values(["segment", "layer", "_timestamp"]).groupby(
            ["segment", "layer"], as_index=False
        )
        .tail(1)
        .sort_values(["segment", "layer"])
        .reset_index(drop=True)
    )
    return df


def scan_wandb_runs_reading(wandb_dir: Optional[Path] = None) -> pd.DataFrame:
    """Scan W&B runs created by the normal (reading) probe trainer."""
    return scan_wandb_runs(wandb_dir, filter_script_suffix="src/probe.py")


def scan_wandb_runs_control(wandb_dir: Optional[Path] = None) -> pd.DataFrame:
    """Scan W&B runs created by the control (steering) probe trainer."""
    return scan_wandb_runs(wandb_dir, filter_script_suffix="src/steering_probe.py")


# -------- Plot helpers --------

def _dropdown_position(position: str) -> Dict:
    """Compute updatemenus position away from the modebar.

    Options: 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    Default 'top-left' to avoid overlapping the modebar (which is top-right).
    """
    position = (position or "top-left").lower()
    if position == "top-right":
        return dict(x=1.0, xanchor="right", y=1.08, yanchor="top")
    if position == "bottom-left":
        return dict(x=0.0, xanchor="left", y=-0.2, yanchor="bottom")
    if position == "bottom-right":
        return dict(x=1.0, xanchor="right", y=-0.2, yanchor="bottom")
    # default top-left
    return dict(x=0.0, xanchor="left", y=1.08, yanchor="top")


def _line_layout(
    fig: go.Figure,
    title: str,
    y_title: str,
    y_range: Optional[Tuple[float, float]] = None,
    legend_bottom: bool = False,
):
    legend = dict(orientation="h", y=1.12, x=0.0)
    if legend_bottom:
        legend = dict(orientation="h", y=-0.25, x=0.0)

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=y_title,
        yaxis=dict(rangemode="tozero") if y_range is None else dict(range=y_range),
        margin=dict(l=40, r=20, t=80, b=60 if legend_bottom else 40),
        legend=legend,
    )


def make_segment_line(
    df: pd.DataFrame,
    metric: str,
    title_prefix: str = "",
    dropdown_position: str = "top-left",
    legend_bottom: bool = False,
) -> go.Figure:
    """Single-metric line chart with a segment dropdown.

    metric in {'test_loss','test_accuracy','train_loss'}
    """
    assert metric in {"test_loss", "test_accuracy", "train_loss"}
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No W&B data found for metric: {metric}")
        return fig

    segments = sorted(df["segment"].unique())

    vals = df[metric].dropna().to_numpy()
    if metric == "test_accuracy":
        y_range = [0.0, 1.02]
    else:
        if vals.size:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            pad = 0.05 * (hi - lo + 1e-9)
            y_range = [max(0.0, lo - pad), hi + pad]
        else:
            y_range = None

    fig = go.Figure()
    for idx, seg in enumerate(segments):
        sub = df[df["segment"] == seg].sort_values("layer")
        layers = sub["layer"].astype(int).tolist()
        values = sub[metric].tolist()
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=values,
                mode="lines+markers",
                name=metric,
                visible=(idx == 0),
            )
        )

    # Dropdown
    buttons = []
    for i, seg in enumerate(segments):
        vis = [False] * len(segments)
        vis[i] = True
        buttons.append(
            dict(
                label=seg,
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": f"{title_prefix}{metric.replace('_', ' ').title()} over layers — {seg}",
                        "showlegend": True,
                    },
                ],
            )
        )

    _line_layout(
        fig,
        f"{title_prefix}{metric.replace('_', ' ').title()} over layers — {segments[0]}",
        metric.replace("_", " ").title(),
        y_range=y_range,
        legend_bottom=legend_bottom,
    )
    pos = _dropdown_position(dropdown_position)
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                **pos,
            )
        ]
    )
    return fig


def make_segment_lines(
    df: pd.DataFrame,
    metrics: List[str],
    title_prefix: str = "",
    dropdown_position: str = "top-left",
    legend_bottom: bool = False,
) -> go.Figure:
    """Multi-metric (e.g., train/test loss) line chart with segment dropdown."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No W&B data found")
        return fig

    segments = sorted(df["segment"].unique())
    colors = {
        "train_loss": "#F59E0B",  # amber
        "test_loss": "#EF4444",  # red
        "test_accuracy": "#10B981",
    }

    vals_list = [df[m] for m in metrics if m in df]
    if len(vals_list):
        vals = pd.concat(vals_list, axis=0).dropna().to_numpy()
        lo, hi = (float(np.nanmin(vals)), float(np.nanmax(vals))) if vals.size else (0.0, 1.0)
        pad = 0.05 * (hi - lo + 1e-9)
        y_range = [max(0.0, lo - pad), hi + pad]
    else:
        y_range = [0.0, 1.0]

    fig = go.Figure()
    for idx, seg in enumerate(segments):
        sub = df[df["segment"] == seg].sort_values("layer")
        layers = sub["layer"].astype(int).tolist()
        for m in metrics:
            ys = sub[m].tolist() if m in sub else [None] * len(layers)
            fig.add_trace(
                go.Scatter(
                    x=layers,
                    y=ys,
                    mode="lines+markers",
                    name=m,
                    line=dict(color=colors.get(m, None)),
                    visible=(idx == 0),
                )
            )

    traces_per_segment = len(metrics)
    total_traces = traces_per_segment * len(segments)
    buttons = []
    for s_idx, seg in enumerate(segments):
        vis = [False] * total_traces
        start = s_idx * traces_per_segment
        for k in range(traces_per_segment):
            vis[start + k] = True
        buttons.append(
            dict(
                label=seg,
                method="update",
                args=[{"visible": vis}, {"title": f"{title_prefix}Loss over layers — {seg}"}],
            )
        )

    _line_layout(fig, f"{title_prefix}Loss over layers — {segments[0]}", "Loss", y_range=y_range, legend_bottom=legend_bottom)
    pos = _dropdown_position(dropdown_position)
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                **pos,
            )
        ]
    )
    return fig



def plot_loss_lines(
    runs_df: Optional[pd.DataFrame] = None,
    wandb_dir: Optional[Path] = None,
    dropdown_position: str = "top-left",
    legend_bottom: bool = False,
) -> go.Figure:
    if runs_df is None:
        runs_df = scan_wandb_runs(wandb_dir)
    return make_segment_lines(
        runs_df,
        metrics=["train_loss", "test_loss"],
        dropdown_position=dropdown_position,
        legend_bottom=legend_bottom,
    )


def plot_accuracy_line(
    runs_df: Optional[pd.DataFrame] = None,
    wandb_dir: Optional[Path] = None,
    dropdown_position: str = "top-left",
    legend_bottom: bool = False,
) -> go.Figure:
    """Return a Plotly Figure: test accuracy over layers with segment dropdown."""
    if runs_df is None:
        runs_df = scan_wandb_runs(wandb_dir)
    return make_segment_line(
        runs_df,
        metric="test_accuracy",
        dropdown_position=dropdown_position,
        legend_bottom=legend_bottom,
    )


def plot_control_loss_lines(
    runs_df: Optional[pd.DataFrame] = None,
    wandb_dir: Optional[Path] = None,
    dropdown_position: str = "top-left",
    legend_bottom: bool = False,
) -> go.Figure:
    """Loss lines for control (steering) probes only."""
    if runs_df is None:
        runs_df = scan_wandb_runs_control(wandb_dir)
    return make_segment_lines(
        runs_df,
        metrics=["train_loss", "test_loss"],
        dropdown_position=dropdown_position,
        legend_bottom=legend_bottom,
    )


def plot_control_accuracy_line(
    runs_df: Optional[pd.DataFrame] = None,
    wandb_dir: Optional[Path] = None,
    dropdown_position: str = "top-left",
    legend_bottom: bool = False,
) -> go.Figure:
    """Accuracy over layers for control (steering) probes only."""
    if runs_df is None:
        runs_df = scan_wandb_runs_control(wandb_dir)
    return make_segment_line(
        runs_df,
        metric="test_accuracy",
        dropdown_position=dropdown_position,
        legend_bottom=legend_bottom,
    )



def _available_layers(artifacts_dir: Path) -> List[int]:
    layers: List[int] = []
    for p in artifacts_dir.glob("layer_*.pt"):
        try:
            num = int(p.stem.split("_")[1])
            layers.append(num)
        except Exception:
            continue
    return sorted(layers)


def _load_classes(task: str, artifacts_dir: Path) -> List[str]:
    classes_path = artifacts_dir / "classes.json"
    if classes_path.exists():
        try:
            return json.loads(classes_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    fallback = {
        "socioeco": ["low", "middle", "high"],
        "religion": ["christianity", "hinduism", "islam"],
        "location": ["europe", "north_america", "east_asia"],
    }
    if task not in fallback:
        raise FileNotFoundError(
            f"Missing classes file at {classes_path} and no fallback for task '{task}'"
        )
    return sorted(fallback[task])


def plot_prompt_probe_heatmap(
    task: str,
    prompt: str,
    artifacts_root: Union[str, Path] = "artifacts",
    model_id: str = "meta-llama/Meta-Llama-3.1-8B",
    layers: Optional[List[int]] = None,
    device: Optional[str] = None,
    model: Optional[object] = None,
    *,
    width: int = 520,
    height: Optional[int] = None,
    height_per_layer: int = 26,
    min_height: int = 420,
    max_height: int = 1200,
    device_map: Optional[str] = None,
    max_tokens: int = 512,
) -> go.Figure:
    """Heatmap (y: layers, x: classes) of probe probabilities for a single prompt.

    - Uses trained probes in `artifacts/{task}/layer_XX.pt`.
    - Applies the task-specific tail truncation before encoding.
    - Returns a Plotly Figure with title: "user's predicted {task} heatmap for prompt: {prompt}".
    """
    task = task.lower().strip()
    artifacts_dir = Path(artifacts_root) / task if artifacts_root else None
    if artifacts_dir is None or not artifacts_dir.exists():
        found = find_artifacts_dir(task)
        if found is None:
            raise FileNotFoundError(
                f"Artifacts not found for task '{task}': {Path(artifacts_root) / task if artifacts_root else 'artifacts/<task>'}"
            )
        artifacts_dir = found

    classes = _load_classes(task, artifacts_dir)
    if layers is None:
        layers = _available_layers(artifacts_dir)
    if not layers:
        raise ValueError(f"No probe layers found in {artifacts_dir}")

    # Lazy imports to avoid heavy deps at module import time
    from nnsight import LanguageModel
    import torch
    from probe import LinearProbe, TAILS, truncate_to_tail, sanitize_conversation

    # Init model
    owns_model = model is None
    # Choose device if not provided (for probe computation)
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    # Decide model device map (where to load the LLM)
    chosen_device_map = device_map
    if chosen_device_map is None:
        chosen_device_map = "cpu" if device == "cpu" else "auto"
    if model is None:
        model = LanguageModel(model_id, device_map=chosen_device_map)

    # Prepare text input (sanitize + tail truncate)
    tail = TAILS.get(task)
    cleaned = sanitize_conversation(prompt)
    safe_text = truncate_to_tail(cleaned, tail) if tail else cleaned
    tokens = model.tokenizer(
        safe_text, return_tensors="pt", truncation=True, max_length=int(max_tokens)
    )

    # Collect last-token activations per requested layer in a single trace
    saved: Dict[int, torch.Tensor] = {}
    def run_trace() -> None:
        saved.clear()
        with torch.no_grad():
            with model.trace(tokens["input_ids"]):
                for l in layers:
                    layer_output = model.model.layers[l].output
                    hidden_states = (
                        layer_output[0] if isinstance(layer_output, tuple) else layer_output
                    )
                    saved[l] = hidden_states[:, -1, :].save()

    try:
        run_trace()
    except Exception as e:
        # Retry on CPU if loading/tracing on GPU/MPS fails due to memory
        if owns_model and chosen_device_map != "cpu":
            try:
                model = LanguageModel(model_id, device_map="cpu")
                tokens = model.tokenizer(
                    safe_text, return_tensors="pt", truncation=True, max_length=int(max_tokens)
                )
                run_trace()
            except Exception:
                raise e
        else:
            raise e

    # Build and apply probes to get probabilities
    d_model = model.config.hidden_size
    n_cls = len(classes)

    # Preload all probes
    probes: Dict[int, LinearProbe] = {}
    for l in layers:
        lp = LinearProbe(d_model, n_cls)
        weights = torch.load(artifacts_dir / f"layer_{l:02d}.pt", map_location=device)
        lp.load_state_dict(weights)
        lp.to(device)
        lp.eval()
        probes[l] = lp

    # z matrix: rows=layers, cols=classes
    z: List[List[float]] = []
    for l in layers:
        act = saved[l].to(device)
        with torch.no_grad():
            logits = probes[l](act)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
        z.append([float(probs[i]) for i in range(n_cls)])

    # Assemble heatmap (layers on y, classes on x)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=classes,
                y=layers,
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                hovertemplate="layer=%{y}<br>class=%{x}<br>prob=%{z}<extra></extra>",
                colorbar=dict(title="prob", thickness=12, len=0.9),
            )
        ]
    )

    # Descriptive title; escape or truncate prompt if very long
    disp_prompt = prompt if len(prompt) <= 200 else (prompt[:200] + "…")
    # Compute a tall layout based on number of layers unless a height was provided
    if height is None:
        calc_h = int(min(max_height, max(min_height, 120 + height_per_layer * len(layers))))
    else:
        calc_h = int(height)

    fig.update_layout(
        title=f"user's predicted {task} heatmap for prompt: {disp_prompt}",
        xaxis_title="Class",
        yaxis_title="Layer",
        xaxis=dict(tickangle=45, automargin=True),
        margin=dict(l=60, r=60, t=80, b=80),
        width=int(width),
        height=calc_h,
    )

    # Cleanup if we created the model (optional)
    _ = owns_model  # placeholder; rely on Python GC
    return fig


def plot_prompt_last_layer_dual_heatmap(
    task: str,
    prompt: str,
    *,
    reading_root: Union[str, Path] = "artifacts",
    control_root: Union[str, Path] = "artifacts_control",
    model_id: str = "meta-llama/Meta-Llama-3.1-8B",
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    layer: Optional[int] = None,
    max_tokens: int = 512,
    width: int = 700,
    height: int = 300,
) -> go.Figure:
    """Compare last-layer class probabilities for a single prompt: reading vs control.

    - Produces a 2-row heatmap with classes on X and rows ['reading','control'].
    - If `layer` is None, uses the last available layer in each artifacts root.
    - Returns a Plotly Figure ready for Jupyter display.
    """
    task = task.lower().strip()

    # Resolve artifact directories and classes
    read_dir = Path(reading_root) / task
    ctrl_dir = Path(control_root) / task
    if not read_dir.exists():
        found = find_artifacts_dir(task)
        if found is None:
            raise FileNotFoundError(f"Reading artifacts not found for task '{task}'")
        read_dir = found
    if not ctrl_dir.exists():
        # Try module-relative control folder
        ctrl_dir = (Path(__file__).resolve().parent.parent / control_root / task).resolve()
        if not ctrl_dir.exists():
            raise FileNotFoundError(f"Control artifacts not found for task '{task}' in {control_root}")

    classes = _load_classes(task, read_dir)

    # Determine layer(s)
    read_layers = _available_layers(read_dir)
    ctrl_layers = _available_layers(ctrl_dir)
    if not read_layers or not ctrl_layers:
        raise ValueError("No probe layers found for reading or control artifacts.")
    read_L = read_layers[-1] if layer is None else int(layer)
    ctrl_L = ctrl_layers[-1] if layer is None else int(layer)

    # Lazy imports for heavy deps
    from nnsight import LanguageModel
    import torch
    from probe import LinearProbe, TAILS, truncate_to_tail, sanitize_conversation

    # Choose device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    chosen_device_map = device_map if device_map is not None else ("cpu" if device == "cpu" else "auto")
    model = LanguageModel(model_id, device_map=chosen_device_map)

    # Prepare prompt
    tail = TAILS.get(task)
    cleaned = sanitize_conversation(prompt)
    safe_text = truncate_to_tail(cleaned, tail) if tail else cleaned
    tokens = model.tokenizer(safe_text, return_tensors="pt", truncation=True, max_length=int(max_tokens))

    # Helper to get probabilities for a single layer from a given artifacts dir
    def layer_probs(art_dir: Path, L: int) -> List[float]:
        d_model = model.config.hidden_size
        n_cls = len(classes)
        probe = LinearProbe(d_model, n_cls)
        weights = torch.load(art_dir / f"layer_{L:02d}.pt", map_location=device)
        probe.load_state_dict(weights)
        probe.to(device)
        probe.eval()
        with torch.no_grad():
            with model.trace(tokens["input_ids"]):
                layer_output = model.model.layers[L].output
                hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                act = hs[:, -1, :].save().to(device)
                logits = probe(act)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
        return [float(probs[i]) for i in range(n_cls)]

    read_probs = layer_probs(read_dir, read_L)
    ctrl_probs = layer_probs(ctrl_dir, ctrl_L)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=[read_probs, ctrl_probs],
                x=classes,
                y=[f"reading (L{read_L})", f"control (L{ctrl_L})"],
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="prob", thickness=12, len=0.7),
                hovertemplate="row=%{y}<br>class=%{x}<br>prob=%{z}<extra></extra>",
            )
        ]
    )
    disp_prompt = prompt if len(prompt) <= 180 else (prompt[:180] + "…")
    fig.update_layout(
        title=f"Last-layer class probabilities — {task} — prompt: {disp_prompt}",
        xaxis_title="Class",
        yaxis_title="Probe type",
        xaxis=dict(tickangle=45, automargin=True),
        margin=dict(l=60, r=60, t=80, b=80),
        width=int(width),
        height=int(height),
    )
    return fig


__all__ = [
    "find_wandb_dir",
    "find_artifacts_dir",
    "scan_wandb_runs",
    "scan_wandb_runs_reading",
    "scan_wandb_runs_control",
    "plot_loss_lines",
    "plot_control_loss_lines",
    "plot_accuracy_line",
    "plot_control_accuracy_line",
    "make_segment_line",
    "make_segment_lines",
    "plot_prompt_probe_heatmap",
    "plot_prompt_last_layer_dual_heatmap",
]


# -------- Confusion Matrix for a probe (test split) --------

def _load_task_texts_and_labels(task: str, data_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Load conversations and labels from data directory matching the task prefix.

    Follows the same filename parsing as training: `<task>_<label>_<id>.txt` where label may contain underscores.
    """
    import glob
    import os

    # Resolve data directory, searching upwards if needed
    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.glob(f"{task}_*.txt")):
        found = find_data_dir(task)
        if found is not None:
            data_path = found
    data_dir = str(data_path)
    conversations: List[str] = []
    labels: List[str] = []

    for file_path in glob.glob(os.path.join(data_dir, "*.txt")):
        filename = os.path.basename(file_path)
        if "_" not in filename:
            continue
        parts = filename.replace(".txt", "").split("_")
        if len(parts) < 2:
            continue
        prefix = parts[0]
        if prefix != task:
            continue
        value = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
        with open(file_path, "r", encoding="utf-8") as f:
            conversation = f.read().strip()
        conversations.append(conversation)
        labels.append(value)

    return conversations, labels


def plot_probe_confusion(
    task: str,
    layer: int,
    *,
    artifacts_root: Union[str, Path] = "artifacts",
    data_dir: Union[str, Path] = "data",
    model_id: str = "meta-llama/Meta-Llama-3.1-8B",
    test_size: float = 0.2,
    random_seed: int = 42,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    max_tokens: int = 512,
    width: int = 520,
    height: int = 460,
) -> go.Figure:
    """Plot a confusion matrix (Plotly heatmap) for a given probe layer.

    - task: one of {'religion','socioeco','location'}
    - layer: layer index for the probe weights
    - Uses a test split generated with the provided seed/ratio to approximate training split
    - Color scheme uses a different scale (e.g., 'Blues')
    """
    # Lazy imports for heavy deps
    from nnsight import LanguageModel
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from probe import LinearProbe, TAILS, truncate_to_tail, sanitize_conversation

    task = task.strip().lower()

    # Locate artifacts and classes
    art_dir = Path(artifacts_root) / task if artifacts_root else None
    if art_dir is None or not art_dir.exists():
        found = find_artifacts_dir(task)
        if found is None:
            raise FileNotFoundError(f"Artifacts not found for task '{task}'")
        art_dir = found
    classes = _load_classes(task, art_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Load dataset and split
    texts, labels_str = _load_task_texts_and_labels(task, data_dir)
    if not texts:
        raise FileNotFoundError(f"No data found for task '{task}' in {data_dir}")
    # Ensure labels are within known classes
    labels_str = [lbl for lbl in labels_str]
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels_str,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels_str,
    )

    # Prepare model and probe
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    chosen_device_map = device_map if device_map is not None else ("cpu" if device == "cpu" else "auto")

    model = LanguageModel(model_id, device_map=chosen_device_map)
    d_model = model.config.hidden_size
    n_cls = len(classes)
    probe = LinearProbe(d_model, n_cls)
    weights = torch.load(art_dir / f"layer_{layer:02d}.pt", map_location=device)
    probe.load_state_dict(weights)
    probe.to(device)
    probe.eval()

    # Helpers to get last-token activation per text
    def last_token_activation(text: str) -> torch.Tensor:
        tail = TAILS.get(task)
        cleaned = sanitize_conversation(text)
        safe_text = truncate_to_tail(cleaned, tail) if tail else cleaned
        tokens = model.tokenizer(safe_text, return_tensors="pt", truncation=True, max_length=int(max_tokens))
        with torch.no_grad():
            with model.trace(tokens["input_ids"]):
                layer_output = model.model.layers[layer].output
                hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                act = hidden_states[:, -1, :].save()
        return act.cpu()

    # Predict for test set
    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    for text, lbl in zip(X_test, y_test):
        act = last_token_activation(text).to(device)
        with torch.no_grad():
            logits = probe(act)
            pred = int(torch.argmax(logits, dim=-1).item())
        y_true_idx.append(class_to_idx.get(lbl, -1))
        y_pred_idx.append(pred)

    # Filter out any unknown labels
    pairs = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if 0 <= t < n_cls]
    if not pairs:
        raise ValueError("No valid label pairs for confusion matrix.")
    y_true_idx, y_pred_idx = zip(*pairs)

    # Confusion matrix with fixed label order
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(n_cls)))

    # Plotly heatmap
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=cm,
                x=classes,
                y=classes,
                colorscale="Blues",
                colorbar=dict(title="count", thickness=12, len=0.9),
                hovertemplate="true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"Confusion Matrix — {task} (layer {layer})",
        xaxis_title="Predicted",
        yaxis_title="True",
        xaxis=dict(tickangle=45, automargin=True),
        margin=dict(l=70, r=40, t=70, b=80),
        width=int(width),
        height=int(height),
    )
    return fig
