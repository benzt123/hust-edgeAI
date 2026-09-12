"""Offline, gold-independent conservative extraction; never modifies original runs."""
import hashlib
import importlib.util
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import full_scoring as v1
VERSION = 'gsm8k_numeric_v2_conservative'
TOKEN = re.compile(r'(?<![\w.])(?:\\(?:dfrac|tfrac|frac)\{[+-]?\d+\}\{[+-]?\d+\}|[+-]?\d+\s+\d+\s*/\s*\d+|[+-]?\d+\s*/\s*[+-]?\d+|[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*\\?%)?(?!\w)')


def number(text):
    text = text.strip()
    mixed = re.fullmatch(r'([+-]?)(\d+)\s+(\d+)\s*/\s*(\d+)', text)
    if mixed:
        sign, whole, a, b = mixed.groups()
        if int(b) == 0 or int(a) >= int(b):
            return None
        return (-1 if sign == '-' else 1) * (int(whole) + Fraction(int(a), int(b)))
    return v1.number(text)


def boxed(text):
    out = []
    for match in re.finditer(r'\\boxed\{', text):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                out.append((text[start:i], i + 1))
                break
    return out


def single(text):
    """One explicit numeric value, allowing prose/units; reject expressions/ranges."""
    direct = number(text)
    if direct is not None:
        return text.strip()
    if re.search(r'\\(?:sqrt|times|div)|[<>≈~]|\b(?:or|between|approximately|about|not|print|return)\b', text, re.I):
        return None
    matches = list(TOKEN.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    outside = text[:match.start()] + text[match.end():]
    if re.search(r'[*/^=+]|(?<!\w)-(?!\w)', outside):
        return None
    return match[0].strip() if number(match[0]) is not None else None


def block(text):
    boxes = boxed(text)
    if boxes and len(text[boxes[-1][1]:].strip()) < 160 and not TOKEN.search(text[boxes[-1][1]:]):
        answer = number(boxes[-1][0])
        if answer is not None:
            return boxes[-1][0]
    value = single(text)
    if value is not None:
        return value
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    tail = lines[-1] if lines else ''
    # An equation must end in an explicit numeric RHS, never evaluate generated code.
    if '=' in tail and not re.search(r'[<>≈]', tail):
        return single(tail.rsplit('=', 1)[1])
    if len(lines) > 1:
        return single(tail)
    return None


def extract(response):
    """Does not accept gold; cannot search for a matching reference number."""
    tags = re.findall(r'<answer>(.*?)</answer>', response, re.S)
    if len(tags) > 1:
        return None, 'ambiguous_answer_tags'
    if tags:
        return block(tags[0]), 'answer_tag'
    if '<answer>' in response:
        return None, 'unclosed_answer_tag'
    boxes = boxed(response)
    if boxes:
        candidate, end = boxes[-1]
        tail = response[end:]
        if len(tail.strip()) < 160 and not TOKEN.search(tail) and number(candidate) is not None:
            return candidate, 'terminal_boxed'
    if '####' in response:
        return block(response.rsplit('####', 1)[1]), 'hash_answer'
    # No hunting through intermediate reasoning: inspect the final nonempty line only.
    lines = [x.strip() for x in response.splitlines() if x.strip()]
    tail = lines[-1] if lines else ''
    tail = re.sub(r'</?(?:answer|think)>', '', tail).strip()
    if len(tail) <= 300:
        return block(tail), 'terminal_line'
    return None, 'no_explicit_answer'


def score(response, gold):
    old = v1.score(response, gold)
    candidate, source = extract(response)
    value = number(candidate) if candidate is not None else None
    reference = number(gold)
    if reference is None:
        raise ValueError('Unsupported gold')
    correct = value is not None and value == reference
    return dict(answer_correct=correct, parsed=value is not None,
                format_ok=old['format_ok'], joint_correct=correct and old['format_ok'],
                extracted_answer=candidate, answer_source=source)

