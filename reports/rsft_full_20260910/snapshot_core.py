"""RSFT 核心练习：先筛选，再统计；

学习顺序：
1. 完成 select_correct_samples 中的 TODO 1～3。
2. 用文件底部的模拟候选检查保留结果。
3. 再完成 summarize_candidates 中的 TODO 4～6（进阶练习）。

真实流程中，模型生成 response，评分器提供判断字段。本文件只处理这些记录，
不加载模型、不计算奖励、不执行训练。保留的 response 应是完整生成解答，
不能替换成标准答案，也不要在这里添加或删除 think/answer 标签。
"""
import random


def select_correct_samples(candidates):
    """筛选出可供下一轮 SFT 学习的回答。

    输入：字典列表，每条候选包含：
        question_id: str，题目的唯一 ID。
        prompt: str，实际用于生成的完整提示词。
        response: str，模型生成的完整解答。
        answer_correct: bool，最终答案是否正确。
        format_ok: bool，回答格式是否合格。
        truncated: bool，生成是否被截断。

    本练习假设输入字段齐全，三个判断字段为真正的 bool，不是字符串。

    输出：字典列表，每条只包含 question_id、prompt、response。
    保留条件：答案正确、格式合格、没有截断，三个条件同时成立。
    维持候选原顺序，不修改输入字典。同一道题可保留多个回答。
    空输入或全部不合格时返回 []。
    """
    selected = []

    for candidate in candidates:
        # TODO 1：读取三个判断字段，计算当前候选是否应该保留。
        keep = candidate["answer_correct"] and candidate["format_ok"] and not candidate["truncated"]

        # TODO 2：不满足条件时跳过当前候选，继续检查下一个。
        if not keep:
            continue
        # TODO 3：为合格候选创建一个新字典，加入 selected。
        # 新字典只保留 question_id、prompt、response，字符串原样保存。
        selected.append({
            "question_id": candidate["question_id"],
            "prompt": candidate["prompt"],
            "response": candidate["response"]
        })

    return selected


def select_up_to_k_per_question(selected, k=2, seed=42):
    """每题最多保留k条合格回答，不足k条全部保留，不补齐。

    输入selected来自select_correct_samples；不在这里重复评分。
    相同输入顺序和seed可复现选择。不同候选即使文本相同也不去重。
    返回新字典，不修改输入；没有合格回答的题自然跳过。
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k 必须是大于等于1的整数")

    # 第一步：按题目分组。每个题目ID对应一个回答列表。
    grouped = {}
    for sample in selected:
        question_id = sample["question_id"]
        if question_id not in grouped:
            grouped[question_id] = []
        grouped[question_id].append(sample)

    # 第二步：超过上限就随机抽取；不足上限不填空、不复制。
    rng = random.Random(seed)
    result = []
    for samples in grouped.values():
        if len(samples) <= k:
            chosen = samples
        else:
            chosen = rng.sample(samples, k)

        # 第三步：把各题选出的回答汇总成一个列表。
        for sample in chosen:
            result.append(sample.copy())

    return result


def summarize_candidates(candidates, selected):
    """进阶练习：区分“回答通过率”和“题目覆盖率”。

    selected 是合格回答，或从合格回答中按每题上限再次筛出的子集。
    对合格回答统计时acceptance_rate表示质量通过率；
    对限制数量后的结果统计时，它表示最终保留率，不能混为一谈。
    返回一个包含以下字段的字典：
        candidate_count: 候选回答总数。
        selected_count: 保留下来的回答总数。
        acceptance_rate: selected_count / candidate_count。
        question_count: 候选中不同 question_id 的数量。
        covered_question_count: 保留结果中不同 question_id 的数量。
        question_coverage: covered_question_count / question_count。

    约定：任何比率的分母为 0 时，该比率返回 0.0。
    比率使用 0～1 的数值，不返回带百分号的字符串。
    示例：两道题共四个候选，第一道题有两个合格回答、第二道题全错。
    回答通过率为 2/4，题目覆盖率为 1/2；保留两条不等于覆盖两道题。
    """
    # TODO 4：统计候选总数和保留总数，计算回答通过率。
    # 提示：len；先判断分母，避免 ZeroDivisionError。
    candidate_count = len(candidates)
    selected_count = len(selected)
    acceptance_rate = selected_count / candidate_count if candidate_count > 0 else 0.0

    # TODO 5：分别统计两个列表中不重复的 question_id，计算题目覆盖率。
    # 提示：set 可以去重；不要用回答数量代替题目数量。
    question_ids = set()
    for candidate in candidates:
        question_ids.add(candidate["question_id"])
    question_count = len(question_ids)
    sel_question_ids = set()
    for sel_q in selected:
        sel_question_ids.add(sel_q["question_id"])
    covered_question_count = len(sel_question_ids)
    covered_rate = covered_question_count / question_count if question_count > 0 else 0.0

    # TODO 6：按文档中的六个字段名构造并返回统计字典。
    return {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "acceptance_rate": acceptance_rate,
        "question_count": question_count,
        "covered_question_count": covered_question_count,
        "question_coverage": covered_rate,
    }


# 以下只是手动检查用的模拟数据，不是真实训练样本，也没有调用评分器。
# A、B 是同一道题的两个合格回答，应全部保留。
EXAMPLE_CANDIDATES = [
    dict(question_id="demo/1", prompt="示例题目一", response="完整解答 A",
         answer_correct=True, format_ok=True, truncated=False),
    dict(question_id="demo/1", prompt="示例题目一", response="完整解答 B",
         answer_correct=True, format_ok=True, truncated=False),
    dict(question_id="demo/2", prompt="示例题目二", response="错误解答 C",
         answer_correct=False, format_ok=True, truncated=False),
    dict(question_id="demo/2", prompt="示例题目二", response="格式错误的解答 D",
         answer_correct=True, format_ok=False, truncated=False),
    dict(question_id="demo/3", prompt="示例题目三", response="被截断的解答 E",
         answer_correct=True, format_ok=True, truncated=True),
]


if __name__ == "__main__":
    result = select_correct_samples(EXAMPLE_CANDIDATES)
    print("筛选结果：", result)
    print("预期：只保留 A、B 两条完整解答，每条包含三个字段。")
    print("空输入结果：", select_correct_samples([]))

    print("统计结果：", summarize_candidates(EXAMPLE_CANDIDATES, result))
    # 预期：5个候选，保留2个，通过率0.4；3道题，覆盖1道，覆盖率1/3。

    # 单独演示数量上限：第一题3条合格回答，第二题只有1条。
    qualified = result + [
        dict(question_id="demo/1", prompt="示例题目一", response="完整解答 F"),
        dict(question_id="demo/2", prompt="示例题目二", response="完整解答 G"),
    ]
    selected = select_up_to_k_per_question(qualified, k=2, seed=42)
    print("每题最多两条：", selected)
    print("预期：第一题随机留下2条，第二题留下1条，共3条。")
