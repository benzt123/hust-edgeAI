"""Self-Play + GRPO 学习核心：先完成四个 TODO，再接采样和训练。

proposed_answer 是出题模型提供的参考答案，不是独立验证过的真值。
本轮沿用 GSM8K 数值任务；符号答案须换匹配的评分器，不能硬套数值评分。
"""
import re
import hashlib


PROBLEM_GEN_PROMPT = """Create one self-contained math word problem solvable with arithmetic.
Provide one final numerical answer. Output exactly two labeled fields:
Problem: [problem, without its solution]
Answer: [final numerical answer]
Generate one problem now:"""


def parse_problem_and_answer(text):
    """TODO 1：解析出题输出，返回 (problem, proposed_answer)。

    接受：Problem: 标签在首个非空行，Answer: 标签在后续行行首。
    标签大小写敏感；题干可以多行；答案必须是单个非空行。
    两个标签各出现一次且顺序正确；不接受额外的前言、重复标签或空字段。
    标签前允许水平空白，首尾空白可 strip；不修改字段内部文本。
    不符合约定返回 (None,None)，不猜测缺失字段。

    建议：strip 全文 → 用 re.MULTILINE 找行首标签及次数 → 检查顺序
    → 切出题干与答案 → 检查非空与答案行数。
    本函数只解析格式，不证明题目可解，也不验证答案数值。
    """
    text = text.strip()
    problem_matches = list (
        re.finditer(r"^[ \t]*Problem:", text, flags=re.MULTILINE)
    )
    answer_matches = list(
        re.finditer(r"^[ \t]*Answer:", text, flags=re.MULTILINE)
    )

    if len(problem_matches) != 1 or len(answer_matches) != 1:
        return None, None

    problem_tag = problem_matches[0]
    answer_tag = answer_matches[0]

    if problem_tag.start() != 0:
        return None, None
    if answer_tag.start() < problem_tag.end():
        return None, None

    problem = text[problem_tag.end():answer_tag.start()].strip()
    proposed_answer = text[answer_tag.end():].strip()

    if not problem or not proposed_answer:
        return None, None

    if len(proposed_answer.splitlines()) != 1:
        return None, None

    return problem, proposed_answer


def prepare_problems(generations, answer_supported, blocked_problem_keys=()):
    """TODO 2：过滤并按题干去重，返回新记录列表，不修改输入。

    generations：每条含 text:str、truncated:bool。
    answer_supported(answer)：外部提供的函数，判断评分器能否支持该答案。
    blocked_problem_keys：禁止加入的题干规范化 key 集合（使用 problem_key）。

    步骤：
    1. 跳过截断、解析失败、answer_supported 为 False 的候选。
    2. 用 problem_key(problem) 去重，同一 key 仅留首次有效题目。
    3. 跳过 blocked_problem_keys 中的题目。
    4. 输出 question_id、problem、proposed_answer 三个字段。
       question_id 使用下方 problem_id(key)；空输入返回 []。

    即使同题附带不同答案，也不要将其当作两道题反复训练；首次保留只是
    简单的去重策略，不代表首次答案正确。正式配套应记录丢弃原因供审查。
    """
    blocked = set(blocked_problem_keys)
    seen = set()
    problems = []
    for generation in generations:
        if generation['truncated']:
            continue
        problem, proposed_answer = parse_problem_and_answer(
            generation['text']
            )
        if problem is None or proposed_answer is None:
            continue
        if not answer_supported(proposed_answer):
            continue
        key = problem_key(problem)
        if key in blocked or key in seen:
            continue
        seen.add(key)
        problems.append({
            'question_id': problem_id(key),
            'problem': problem,
            'proposed_answer': proposed_answer,
        })
    return problems


def format_solve_prompt(problem, template):
    """TODO 3：只把题干放进现有解题模板，返回 str。

    template 必须包含且只包含一次字面量 {question}，否则抛 ValueError。
    problem.strip() 为空时抛 ValueError。
    使用 replace 替换该占位符，保留 problem 原文本；模板其他内容不变。
    不使用 str.format，以免数学公式中的其他大括号被当成占位符。
    函数不接收 proposed_answer，防止把参考答案直接泄漏给 solver。
    """
    if not problem.strip():
        raise ValueError("problem 不能为空")
    if template.count("{question}") != 1:
        raise ValueError("template 必须包含且只包含一次 {question}")
    return template.replace("{question}", problem)


def build_reward_groups(problems, candidates, grade, group_size):
    """TODO 4：将完整同题候选组转成 GRPO 输入；返回组记录列表。

    problems：TODO 2 的输出，question_id 不可重复。
    candidates 每条含 question_id、candidate_index、response、truncated。
    candidate_index 应为 0..group_size-1；每题必须有完整一组，不可缺失或重复。
    grade(response, proposed_answer) 返回 answer_correct、format_ok、parsed 三个 bool。

    步骤：
    1. group_size 必须为 >=2 的整数（不接受 bool）；否则抛 ValueError。
    2. 按 question_id 和 candidate_index 收集。未知题目、重复 ID/index、
       非法 index、缺少候选都抛 ValueError，不能把缺失回答补成错误回答。
    3. 按 problems 的顺序遍历，每题候选按 index 升序排序。
    4. 调用 grade，reward=1.0 当且仅当答案匹配、格式合格、成功解析且未截断，
       否则为0.0。保留所有候选，包括错误、截断以及奖励全相同的组。
    5. 每组返回：question_id、responses（完整原文本列表）、rewards（float列表）。
       空 problems 和空 candidates 返回 []；不修改输入。

    注意：评分中的 answer_correct 仅表示与 proposed_answer 相符。
    这里不做 tokenization，不算 advantage；后面直接复用 GRPO 核心。
    """
    if (
        isinstance(group_size, bool) or
        not isinstance(group_size, int) or
        group_size < 2
    ):
        raise ValueError("group_size 必须是 >=2 的整数")
    problem_map = {}
    grouped = {}

    for problem in problems:
        qid = problem['question_id']
        if qid in problem_map:
            raise ValueError(f"重复的 question_id: {qid}")
        problem_map[qid] = problem
        grouped[qid] = [None] * group_size

    for candidate in candidates:
        qid = candidate['question_id']
        index = candidate['candidate_index']
        if qid not in problem_map:
            raise ValueError(f"未知的 question_id: {qid}")
        if (            
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < group_size
        ):
            raise ValueError(f"非法的 candidate_index: {index}")
        if grouped[qid][index] is not None:
            raise ValueError(f"重复的 candidate_index {index} for question_id {qid}")
        grouped[qid][index] = candidate

    results = []
    for problem in problems:
        qid = problem['question_id']
        proposed_answer = problem['proposed_answer']
        indexed_candidates = grouped[qid]
        if any(candidate is None for candidate in indexed_candidates):
            raise ValueError(f"缺少候选回答 for question_id {qid}")
        responses = []
        rewards = []
        for index in range(group_size):
            candidate = indexed_candidates[index]
            response = candidate['response']
            score = grade(response, proposed_answer)
            accepted = (
                score['answer_correct'] and
                score['format_ok'] and
                score['parsed'] and
                not candidate['truncated']
            )
            responses.append(response)
            rewards.append(1.0 if accepted else 0.0)
        results.append({
            'question_id': qid,
            'responses': responses,
            'rewards': rewards,
        })
    return results


def problem_key(problem):
    """只折叠空白，保留大小写及数学符号；不能识别语义重复或改写泄漏。"""
    return ' '.join(problem.split())


def problem_id(key):
    return 'selfplay/' + hashlib.sha256(key.encode('utf-8')).hexdigest()
