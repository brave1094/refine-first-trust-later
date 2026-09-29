"""
02_preprocess/noise_rule/base.py
─────────────────────────────────
Noise rule base (one-hot marking + 2-tier override).

Philosophy: marking 'provides information'; it does not 'remove'.
  → every applicable rule is marked (0/1 column). Whether to drop the session is decided downstream.

2 tiers:
  common (noise_rule_all)  : rules without class references. Same marking on all datasets.
  class (noise_rule_{ds}): rules that need to know the class, e.g. "~ of some class".
       - either 'add' new codes,
       - or 'replace' common codes listed in overrides (class-aware version).

Each rule class implements mark(df) -> [(rule_code, bool Series), ...].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

import pandas as pd

# Overall table order (column sort key). Sorted in this order whether from common or class rules.
# Order of the 30 common (class-independent) rules. Per-dataset Eg_n are appended after these automatically.
ORDER_COMMON = [
    "Aa_1", "Aa_2", "Aa_3", "Aa_4", "Aa_5", "Aa_6", "Aa_7",
    "Bb_1", "Bb_2", "Bb_3", "Bb_4", "Bb_5", "Bb_6",
    "Bb_7", "Bb_8", "Bb_9", "Bb_10", "Bb_11", "Bb_12",
    "Bb_13", "Bb_14", "Bb_15", "Bb_16", "Bb_17", "Bb_18",
    "Bb_19", "Bb_20", "Bb_21", "Bb_22", "Bb_23",
    "Bb_24", "Bb_25", "Bb_26", "Bb_27", "Bb_28", "Bb_29",
    "Cc_1", "Cc_2", "Cc_3", "Cc_4", "Cc_5",
    "Cd_1", "Cd_2", "Cd_3", "Cd_4",
    "De_1", "De_2", "De_3", "De_4",
    "Df_1",
]


class BaseNoiseRules(ABC):
    dataset_name: str = ""
    overrides: set = set()        # which common codes the class rules replace
    # Default refinement rule list for this dataset, applied automatically when
    # 05_make_filelist builds denoised without --new_rule. Sessions marked 1 on these rules are removed.
    default_clean_rules: list = []

    @abstractmethod
    def mark(self, df: pd.DataFrame) -> list:
        """Returns [(rule_code, bool mask), ...]."""
        raise NotImplementedError

    @staticmethod
    def _scol(df: pd.DataFrame, name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str).str.lower().str.strip()
        return pd.Series("", index=df.index)

    @staticmethod
    def _scol_raw(df: pd.DataFrame, name: str) -> pd.Series:
        """Unlike _scol, preserves case.
           (needed for rules whose signature is 'uppercase only', e.g. HULK URI)"""
        if name in df.columns:
            return df[name].fillna("").astype(str).str.strip()
        return pd.Series("", index=df.index)

    @staticmethod
    def _num(df: pd.DataFrame, name: str, default: float = 0) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    @staticmethod
    def _first_octet(ip: str) -> int:
        try:
            return int(ip.split(".")[0])
        except (ValueError, IndexError, AttributeError):
            return -1


def merge_marks(common_marks: list, class_marks: list, overrides: set) -> list:
    """
    Merge common + class marks and sort in table order (ORDER_24).
      - for codes in overrides, drop the common one and use the class one.
      - new codes added by the class also go to their table-order position.
    """
    merged = {}
    for code, m in common_marks:
        if code in overrides:
            continue
        merged[code] = m
    for code, m in class_marks:
        merged[code] = m
    ordered = [(c, merged[c]) for c in ORDER_COMMON if c in merged]
    # codes not in ORDER_COMMON (per-dataset Eg_n) are appended in emit order
    for c in merged:
        if c not in ORDER_COMMON:
            ordered.append((c, merged[c]))
    return ordered


def marks_to_onehot(marks: list, index) -> "OrderedDict":
    out: "OrderedDict[str, pd.Series]" = OrderedDict()
    for code, mask in marks:
        m = mask.reindex(index).fillna(False).astype(bool)
        out[code] = (out[code].astype(bool) | m).astype(int) if code in out else m.astype(int)
    return out


def onehot_report(onehot: "OrderedDict", index) -> str:
    total = len(index)
    if onehot:
        mat = pd.concat([s for s in onehot.values()], axis=1)
        any_noise = mat.astype(bool).any(axis=1)
        multi = int((mat.sum(axis=1) >= 2).sum())
    else:
        any_noise = pd.Series(False, index=index)
        multi = 0
    noisy = int(any_noise.sum())
    clean = total - noisy
    lines = [
        f"Total sessions  : {total:,}",
        f"Noise sessions  : {noisy:,} ({noisy / total * 100:.1f}%)  ※ at least one rule column is 1",
        f"Clean sessions  : {clean:,} ({clean / total * 100:.1f}%)",
        f"Multi-label     : {multi:,} (2+ rules on one session)",
        f"Rule columns    : {len(onehot)}",
        "",
        "[Sessions per rule column]",
    ]
    for code, s in onehot.items():
        lines.append(f"{code:8s}  {int(s.sum()):,}")
    return "\n".join(lines)