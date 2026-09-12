"""纯 CPU/标准库检查，不加载模型。未填写 TODO 时预期报错。"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import (parse_problem_and_answer,prepare_problems,format_solve_prompt,
                  build_reward_groups,problem_key)


class CoreChecks(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(parse_problem_and_answer('Problem: A has 2.\nThen gets 3.\nAnswer: 5'),
                         ('A has 2.\nThen gets 3.','5'))
        for text in ['', 'Problem: X', 'Problem:\nAnswer: 5', 'Answer: 5\nProblem: X',
                     'Hello\nProblem: X\nAnswer: 5', 'Problem: X\nAnswer: 5\nAnswer: 6',
                     'Problem: X\nAnswer: 5\nExplanation', 'Problem: X\nProblem: Y\nAnswer: 5']:
            self.assertEqual(parse_problem_and_answer(text),(None,None))

    def test_filter_and_deduplicate(self):
        records=[dict(text='Problem: A  has 2?\nAnswer: 2',truncated=False),
                 dict(text='Problem: A has 2?\nAnswer: 3',truncated=False),
                 dict(text='Problem: Unsupported?\nAnswer: x',truncated=False),
                 dict(text='Problem: Cut?\nAnswer: 2',truncated=True)]
        before=copy.deepcopy(records)
        result=prepare_problems(records,lambda a:a.isdigit())
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['proposed_answer'],'2')
        self.assertEqual(records,before)
        self.assertEqual(prepare_problems(records,lambda a:a.isdigit(),{problem_key('A has 2?')}),[])
        self.assertEqual(prepare_problems([],lambda a:True),[])

    def test_prompt(self):
        self.assertEqual(format_solve_prompt('What is 2+3?','Formula {x}: {question}'),
                         'Formula {x}: What is 2+3?')
        for template in ['none','{question} {question}']:
            with self.assertRaises(ValueError):format_solve_prompt('X',template)

    def test_rewards_keep_wrong_and_equal_groups(self):
        problems=[dict(question_id='q',problem='X',proposed_answer='5')]
        candidates=[dict(question_id='q',candidate_index=1,response='wrong',truncated=False),
                    dict(question_id='q',candidate_index=0,response='5',truncated=False)]
        def grade(response,answer):
            return dict(answer_correct=response==answer,format_ok=True,parsed=True)
        before=copy.deepcopy(candidates)
        groups=build_reward_groups(problems,candidates,grade,2)
        self.assertEqual(groups,[dict(question_id='q',responses=['5','wrong'],rewards=[1.,0.])])
        self.assertEqual(candidates,before)
        for row in candidates:row['truncated']=True
        self.assertEqual(build_reward_groups(problems,candidates,grade,2)[0]['rewards'],[0.,0.])
        with self.assertRaises(ValueError):build_reward_groups(problems,candidates[:1],grade,2)
        with self.assertRaises(ValueError):build_reward_groups(problems,[candidates[0]]*2,grade,2)
        with self.assertRaises(ValueError):build_reward_groups([],candidates,grade,2)


if __name__=='__main__':unittest.main()
