"""Offline, gold-independent conservative extraction; never modifies original runs."""
import hashlib
import importlib.util
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('numeric_v1', ROOT / 'SFT/full_scoring.py')
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
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


def load(path):
    text = path.read_text(encoding='utf-8-sig')
    return [json.loads(x) for x in text.split('\n') if x.strip()] if path.suffix == '.jsonl' else json.loads(text)


def main():
    report_root = ROOT / 'test/hust-edgeAI/reports'
    out = report_root / 'unified_rescore_20260911'
    out.mkdir(exist_ok=True)
    groups = {
        'base_test': [ROOT / 'SFT/results_20260908/test_base/predictions.jsonl'],
        'sft_test': [ROOT / 'SFT/results_20260908/test_best/predictions.jsonl'],
        'rsft_test': [report_root / 'rsft_full_20260910/test_best/predictions.jsonl'],
        'rsft_validation_before': [report_root / 'rsft_full_20260910/before/predictions.jsonl'],
        'trial_validation_before': [report_root / 'rsft_trial_20260910/before/predictions.jsonl'],
        'trial_validation_after': [report_root / 'rsft_trial_20260910/after/predictions.jsonl'],
        'trial_candidates': [report_root / 'rsft_trial_20260910/candidates.jsonl'],
        'full_candidates': sorted((report_root / 'rsft_full_20260910/chunks').glob('*.json')),
    }
    remote = out / 'remote_originals'
    if remote.exists():
        groups['historical_gsm8k_50'] = [remote / 'runs/20260905_194346_427196/results.jsonl']
        groups['smoke_candidates'] = sorted((remote / 'chunks').glob('*.json'))
    metrics, manifest, test_sets = {}, [], {}
    for name, paths in groups.items():
        rows = []
        for path in paths:
            records = load(path)
            manifest.append(dict(group=name, path=str(path.relative_to(ROOT)), records=len(records), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            rows.extend(records)
        counts = Counter()
        rescored = []
        for row in rows:
            if 'ground_truth' in row:
                row = dict(row, gold=row['ground_truth'], id=row['sample_id'])
            old, new = v1.score(row['response'], row['gold']), score(row['response'], row['gold'])
            for key in ('answer_correct', 'parsed', 'format_ok', 'joint_correct'):
                counts['old_' + key] += old[key]
                counts['new_' + key] += new[key]
            counts['newly_correct'] += new['answer_correct'] and not old['answer_correct']
            counts['lost_correct'] += old['answer_correct'] and not new['answer_correct']
            counts['truncated'] += bool(row.get('truncated'))
            rescored.append(dict(row, original_score=old, unified_score=new))
        metrics[name] = dict(n=len(rows), **counts)
        with (out / (name + '.jsonl')).open('w', encoding='utf-8') as f:
            for row in rescored:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        if name.endswith('_test'):
            test_sets[name] = rescored
            cases = [x for x in rescored if x['original_score']['answer_correct'] != x['unified_score']['answer_correct'] or not x['unified_score']['parsed']]
            (out / (name + '_audit.json')).write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    signatures = [[(x['id'], x['prompt'], x['gold']) for x in test_sets[name]] for name in ('base_test', 'sft_test', 'rsft_test')]
    assert signatures[0] == signatures[1] == signatures[2], 'Test sets differ'
    (out / 'summary.json').write_text(json.dumps(dict(version=VERSION, metrics=metrics, manifest=manifest), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
