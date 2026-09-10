"""RSFT 核心练习：先筛选，再统计；

学习顺序：
1. 完成 select_correct_samples 中的 TODO 1～3。
2. 用文件底部的模拟候选检查保留结果。
3. 再完成 summarize_candidates 中的 TODO 4～6（进阶练习）。

真实流程中，模型生成 response，评分器提供判断字段。本文件只处理这些记录，
不加载模型、不计算奖励、不执行训练。保留的 response 应是完整生成解答，
不能替换成标准答案，也不要在这里添加或删除 think/answer 标签。
"""


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
        # 提示：使用 and 组合条件，注意 truncated 的真假含义。
        # 写好这一段后，删除下面这行占位异常。
        raise NotImplementedError("TODO 1：判断候选是否满足保留条件")

        # TODO 2：不满足条件时跳过当前候选，继续检查下一个。
        # 提示：思考 continue 与 return 在循环里的区别。

        # TODO 3：为合格候选创建一个新字典，加入 selected。
        # 新字典只保留 question_id、prompt、response，字符串原样保存。
        # 提示：不要直接 append(candidate)，也不要修改 candidate。

    return selected


def summarize_candidates(candidates, selected):
    """进阶练习：区分“回答通过率”和“题目覆盖率”。

    selected 必须来自 select_correct_samples(candidates)。
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

    # TODO 5：分别统计两个列表中不重复的 question_id，计算题目覆盖率。
    # 提示：set 可以去重；不要用回答数量代替题目数量。

    # TODO 6：按文档中的六个字段名构造并返回统计字典。
    raise NotImplementedError("TODO 4～6：统计回答通过率和题目覆盖率")


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

    # 完成 TODO 4～6 后，取消下一行注释再运行：
    # print("统计结果：", summarize_candidates(EXAMPLE_CANDIDATES, result))
    # 预期：5个候选，保留2个，通过率0.4；3道题，覆盖1道，覆盖率1/3。
