# -*- coding: utf-8 -*-

"""Language classification helpers shared by subtitle processing modules."""

CJK_LANGUAGE_CODES = {"zho", "jpn", "kor", "chi", "zh", "ja", "ko"}


def normalize_language_code(language_code: str) -> str:
    normalized = (language_code or "").strip().lower().replace("_", "-")
    primary_code = normalized.split("-", 1)[0]
    return primary_code if len(primary_code) <= 2 else primary_code[:3]


def is_cjk_language(language_code: str) -> bool:
    return normalize_language_code(language_code) in CJK_LANGUAGE_CODES
