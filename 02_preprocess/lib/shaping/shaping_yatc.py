#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shaping_yatc.py — YaTC (AAAI 2023) MFR input conversion.

5 packets × (header 80B + payload 240B) = 1600B → (40,40) uint8.
The anonymization of the paper (zeroing ports + IP masking) is not in the original public code,
so this framework reproduces it with the --ip_mask --port_mask flags (INTEGRATION_SPEC.md).
"""
from ._bytes1600 import make_mfr, MfrWriter, build_mfr_from_views


def make_sample(packets, meta, opt):
    return make_mfr(packets, meta, opt)


def build_from_views(views, meta, opt):      # multi-model path (parse shared once)
    return build_mfr_from_views(views, meta, opt)


Writer = MfrWriter
