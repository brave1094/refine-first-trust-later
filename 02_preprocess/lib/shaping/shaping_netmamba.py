#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shaping_netmamba.py — NetMamba (ICNP 2024) input transformation.

5 packets × (header 80B + payload 240B) = 1600B → (40,40) uint8 (semantically a 1-D sequence).
The original preprocessing by default masks only IP addresses to 0.0.0.0 (ports kept) →
enable --ip_mask to reproduce the original (INTEGRATION_SPEC.md).
"""
from ._bytes1600 import make_mfr, MfrWriter, build_mfr_from_views


def make_sample(packets, meta, opt):
    return make_mfr(packets, meta, opt)


def build_from_views(views, meta, opt):      # multi-model path (one shared parse)
    return build_mfr_from_views(views, meta, opt)


Writer = MfrWriter
