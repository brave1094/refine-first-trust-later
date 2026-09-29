"""
02_preprocess/noise_rule/__init__.py
─────────────────────────────────────
Noise rule registry (2 tiers, mark-all).

  common rules : noise_rule_all.AllNoiseRules  — 22 rules without class references, identical marking on all datasets.
  class rules: noise_rule_{dataset}.py      — class-dependent additions/overrides (Cc_4/Cd_1 override, Bb etc.).

Scans NN_noise_rule_*.py (numeric prefix) files except noise_rule_all.py and
registers BaseNoiseRules subclasses in the class-rule registry by dataset_name.
(file names containing dots are loaded safely via importlib.util)

Public API:
  get_common_rules()         -> AllNoiseRules()
  get_class_rules(dataset)   -> class-rule instance or None
  has_class_rules(dataset)   -> bool
  class_datasets()           -> list of datasets that have class rules
"""

import importlib.util
import inspect
import sys
from pathlib import Path

from .base import BaseNoiseRules

_PKG_DIR = Path(__file__).resolve().parent
_CLASS_REGISTRY: dict = {}


def _load_rule_file(path: Path) -> None:
    safe_stem = path.stem.replace(".", "_")
    module_name = f"{__name__}._auto_{safe_stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseNoiseRules) and obj is not BaseNoiseRules:
            name = getattr(obj, "dataset_name", None)
            if name and name != "all":
                _CLASS_REGISTRY[name] = obj


def _discover() -> None:
    for path in sorted(_PKG_DIR.glob("*noise_rule_*.py")):
        if path.stem == "noise_rule_all":
            continue
        try:
            _load_rule_file(path)
        except Exception as e:
            print(f"[noise_rule] skip {path.name}: {type(e).__name__}: {e}",
                  file=sys.stderr)


_discover()


# Official dataset list (in directory-number order) — '--dataset all' expands to this.
ALL_DATASETS = [
    "ustc16", "cic17", "cic18", "iot23", 
    "vpn16", "tor16", "tls1.3", "cispec", 
]


def get_common_rules():
    from .noise_rule_all import AllNoiseRules
    return AllNoiseRules()


def get_class_rules(dataset: str):
    cls = _CLASS_REGISTRY.get(dataset)
    return cls() if cls is not None else None


def get_default_clean_rules(dataset: str) -> list:
    """Default refinement rule list of a dataset (applied automatically by 05 when --new_rule is not given).
    Read from the default_clean_rules attribute of the per-dataset py. Empty list if absent."""
    cls = _CLASS_REGISTRY.get(dataset)
    if cls is not None:
        return list(getattr(cls, 'default_clean_rules', []) or [])
    return []


def has_class_rules(dataset: str) -> bool:
    return dataset in _CLASS_REGISTRY


def class_datasets() -> list:
    return sorted(_CLASS_REGISTRY.keys())
