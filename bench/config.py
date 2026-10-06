"""Config loading, run expansion, and provenance recorded in every result file."""
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Run:
    """One unit of work: a model, a task, one condition, one seed."""
    model: dict
    task: str
    condition: dict
    seed: int

    @property
    def result_name(self):
        parts = [f"{k}-{v}" for k, v in sorted(self.condition.items())]
        return f"{self.model['name']}__{'_'.join(parts)}__seed{self.seed}.json"


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())

    for key in ("run_name", "models", "tasks", "seeds"):
        if not cfg.get(key):
            raise ValueError(f"config is missing required key: {key}")
    for model in cfg["models"]:
        if "name" not in model or "model_id" not in model:
            raise ValueError(f"each model needs 'name' and 'model_id': {model}")

    cfg.setdefault("output_dir", "results")
    return cfg


def expand_runs(cfg):
    """Cross the models, tasks, per-task condition lists, and seeds."""
    from bench.registry import get_adapter

    runs = []
    invalid = []
    for model in cfg["models"]:
        for task_cfg in cfg["tasks"]:
            adapter = get_adapter(task_cfg["task"])
            for condition in adapter.expand_conditions(task_cfg):
                n = condition.get("n_images")
                hi = adapter.max_n_images(condition) if hasattr(adapter, "max_n_images") else None
                lo = adapter.min_n_images(condition) if hasattr(adapter, "min_n_images") else None
                if n is not None and ((hi is not None and n > hi) or (lo is not None and n < lo)):
                    invalid.append(f"{task_cfg['task']} {condition} (allowed n_images: {lo or 1}..{hi})")
                    continue
                for seed in cfg["seeds"]:
                    runs.append(Run(model=model, task=task_cfg["task"], condition=condition, seed=seed))

    # Tasks would otherwise clamp silently and write results labelled with the wrong size.
    if invalid:
        raise ValueError("config asks for sizes the datasets cannot supply:\n  " + "\n  ".join(dict.fromkeys(invalid)))
    return runs


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def provenance(cfg):
    """Recorded in every result file so a run can be traced back to its code and config."""
    import torch
    import transformers

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "config_hash": config_hash(cfg),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
