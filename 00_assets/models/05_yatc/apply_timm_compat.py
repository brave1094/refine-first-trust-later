# -*- coding: utf-8 -*-
"""Add our only change to the upstream YaTC code (NSSL-SJTU/YaTC @ a220b75): a forward() shim in the fine-tuning
model class of models_YaTC.py. timm>=0.9 changed VisionTransformer.forward to call forward_features(x, attn_mask=...)
and forward_head(); the shim restores the timm-0.3.2 behaviour (features -> head) that YaTC was written for.
The four lines are inserted right after the first `return outcome` (end of forward_features). Idempotent."""
import os, sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_YaTC.py")
SHIM = [
    "",
    "    # timm>=0.9 compatibility shim (added by refine-first-trust-later): newer timm VisionTransformer.forward calls",
    "    # forward_features(x, attn_mask=...)/forward_head; restore the timm-0.3.2 behaviour (features -> head) explicitly.",
    "    def forward(self, x, **kwargs):",
    "        x = self.forward_features(x)",
    "        return self.head(x)",
]
if not os.path.exists(PATH):
    sys.exit(f"[FAIL] {PATH} not found — run 00_assets/models/setup_models.sh first")
lines = open(PATH, encoding="utf-8").read().split("\n")
if any("timm>=0.9 compatibility shim" in l for l in lines):
    print("[ok] YaTC timm shim already present")
    sys.exit(0)
i = next((k for k, l in enumerate(lines) if l.strip() == "return outcome"), None)
if i is None:
    sys.exit("[FAIL] anchor `return outcome` not found in models_YaTC.py (unexpected upstream version)")
lines[i + 1:i + 1] = SHIM
open(PATH, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("[ok] YaTC timm shim added after line", i + 1)
