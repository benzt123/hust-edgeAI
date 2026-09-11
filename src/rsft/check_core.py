"""无需GPU：验证筛选条件、空输入、题目去重和不修改原数据。"""
import copy
import itertools
import unittest
from core import select_correct_samples, summarize_candidates, EXAMPLE_CANDIDATES
from core import select_up_to_k_per_question


class CoreTests(unittest.TestCase):
    def test_per_question_cap(self):
        rows=[dict(question_id=q,prompt=q,response=str(i))
              for q,n in [('one',1),('two',2),('four',4)] for i in range(n)]
        original=copy.deepcopy(rows)
        selected=select_up_to_k_per_question(rows,k=2,seed=42)
        self.assertEqual([sum(r['question_id']==q for r in selected)
                          for q in ['one','two','four']], [1,2,2])
        self.assertEqual(len({(r['question_id'],r['response']) for r in selected}),5)
        self.assertEqual(selected,select_up_to_k_per_question(rows,k=2,seed=42))
        selected[0]['response']='changed'
        self.assertEqual(rows,original)
        self.assertEqual(select_up_to_k_per_question([]),[])
        for k in [0,-1,1.5,True]:
            with self.assertRaises(ValueError): select_up_to_k_per_question(rows,k=k)

    def test_all_boolean_combinations(self):
        for correct, formatted, truncated in itertools.product([False, True], repeat=3):
            row=dict(question_id='q',prompt='p',response='r',answer_correct=correct,
                     format_ok=formatted,truncated=truncated)
            self.assertEqual(len(select_correct_samples([row])), int(correct and formatted and not truncated))

    def test_example_and_no_mutation(self):
        original=copy.deepcopy(EXAMPLE_CANDIDATES)
        selected=select_correct_samples(EXAMPLE_CANDIDATES)
        self.assertEqual([x['response'] for x in selected], ['完整解答 A','完整解答 B'])
        self.assertEqual(set(selected[0]), {'question_id','prompt','response'})
        self.assertEqual(summarize_candidates(EXAMPLE_CANDIDATES,selected),
            dict(candidate_count=5,selected_count=2,acceptance_rate=0.4,
                 question_count=3,covered_question_count=1,question_coverage=1/3))
        selected[0]['response']='changed'
        self.assertEqual(EXAMPLE_CANDIDATES, original)

    def test_empty_and_all_rejected(self):
        self.assertEqual(select_correct_samples([]), [])
        self.assertEqual(summarize_candidates([],[]), dict(candidate_count=0,selected_count=0,
            acceptance_rate=0.,question_count=0,covered_question_count=0,question_coverage=0.))
        rows=EXAMPLE_CANDIDATES[2:]
        self.assertEqual(select_correct_samples(rows), [])
        self.assertEqual(summarize_candidates(rows,[])['question_coverage'], 0.)


if __name__=='__main__': unittest.main(verbosity=2)
