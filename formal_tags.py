"""Formal run tag helpers (model profile suffix)."""
from __future__ import annotations

import os


def model_tag_suffix() -> str:
    """e.g. '_m4b9b' or '_m9b' or ''."""
    suf = os.environ.get("FORMAL_TAG_SUFFIX", "").strip()
    if suf:
        return suf if suf.startswith("_") else f"_{suf}"
    prof = os.environ.get("FORMAL_MODEL_PROFILE", "").strip()
    if not prof:
        return ""
    return prof if prof.startswith("_") else f"_{prof}"


def with_model_suffix(tag: str) -> str:
    suf = model_tag_suffix()
    if not suf:
        return tag
    if tag.endswith(suf):
        return tag
    return f"{tag}{suf}"


def strip_model_suffix(tag: str) -> str:
    for suf in ("_m4b9b", "_m9b"):
        if tag.endswith(suf):
            return tag[: -len(suf)]
    return tag
