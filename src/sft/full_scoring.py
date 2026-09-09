"""GSM8K 数值验证 v1：格式和答案分开；不调用旧 SymPy 评分器。"""
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

VERSION = 'gsm8k_numeric_v1'


def number(text):
    # 仅接受完整数值表达式，不从长段文字中挑一个碰巧正确的数字。
    text = text.strip().replace(r'\%', '%')
    if len(text)>256:
        return None
    exponent=re.search(r'[eE]([+-]?\d+)',text)
    if exponent and (len(exponent[1])>5 or abs(int(exponent[1]))>1000):
        return None
    if text.startswith(r'\boxed{') and text.endswith('}'):
        text = text[7:-1].strip()
    if text.startswith('$') and text.endswith('$'):
        text = text[1:-1].strip()
    percent = text.endswith('%')
    if percent:
        text = text[:-1].strip()
    text = text.replace('−', '-')
    if ',' in text:
        if not re.fullmatch(r'[+-]?\d{1,3}(,\d{3})+(\.\d+)?', text):
            return None
        text = text.replace(',', '')
    frac = re.fullmatch(r'\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}', text)
    try:
        if frac:
            value = Fraction(int(frac[1]), int(frac[2]))
        elif re.fullmatch(r'[+-]?\d+\s*/\s*[+-]?\d+', text):
            a, b = text.split('/')
            value = Fraction(int(a), int(b))
        elif re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
            value = Fraction(Decimal(text))
        else:
            return None
        return value / 100 if percent else value
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError):
        return None


def score(response, gold):
    pattern = r'(?:<think>)?(?:(?!</?think>|</?answer>).)*</think>\s*<answer>((?:(?!</?think>|</?answer>).)+)</answer>\s*'
    # DOTALL；必须按顺序出现唯一一对答案标签，且不能带尾随内容。
    format_ok = re.fullmatch(pattern, response, re.S) is not None
    matches = re.findall(r'<answer>(.*?)</answer>', response, re.S)
    candidate = matches[0].strip() if len(matches) == 1 else None
    source = 'answer_tag' if candidate is not None else None
    if candidate is None and '<answer>' not in response and '</answer>' not in response:
        # 没有 answer 标签时，只接受末尾 boxed 或整段纯数值作为明确最终答案。
        pos = response.rfind(r'\boxed{')
        if pos >= 0 and response[pos:].strip().endswith('}'):
            candidate = response[pos:].strip()
            source = 'terminal_boxed'
        elif number(response) is not None:
            candidate, source = response.strip(), 'whole_numeric'
    reference = number(gold)
    if reference is None:
        raise ValueError(f'标准答案不是支持的数值：{gold!r}')
    prediction = number(candidate) if candidate is not None else None
    correct = prediction is not None and prediction == reference
    return dict(answer_correct=correct, format_ok=format_ok, joint_correct=correct and format_ok,
                parsed=prediction is not None, answer_source=source, extracted_answer=candidate)
