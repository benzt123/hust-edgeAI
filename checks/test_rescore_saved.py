"""Regression tests for saved-answer extraction, including false-positive traps."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('rescore', Path(__file__).resolve().parents[1] / 'src/common/rescore_saved.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExtractionTests(unittest.TestCase):
    def test_supported_final_answers(self):
        for response, gold in [
            (r'The profit is \(\boxed{70000}\).', '70000'),
            ('</think><answer>26 pieces</answer>', '26'),
            ('</think><answer>So, Siobhan has 23 jewels.</answer>', '23'),
            (r'Final: \boxed{\frac{2}{3}}.', '2/3'),
            ('</think><answer>-2 1/3</answer>', '-7/3'),
            ('Final answer: $1,200.', '1200'),
            ('Work: 4 + 5\nTotal = 4 + 5 = 9.', '9'),
        ]:
            with self.subTest(response=response):
                self.assertTrue(m.score(response, gold)['answer_correct'])

    def test_no_gold_hunting_or_expression_stripping(self):
        for response, gold in [
            ('Intermediate answer 23\nFinal answer: 24', '23'),
            ('<answer>23 or 24</answer>', '23'),
            ('<answer>23</answer><answer>24</answer>', '23'),
            ('<answer>23 + 1</answer>', '23'),
            (r'<answer>\sqrt{23}</answer>', '23'),
            ('<answer>23%</answer>', '23'),
            ('<answer>23', '23'),
            ('There is no answer.', '23'),
        ]:
            with self.subTest(response=response):
                self.assertFalse(m.score(response, gold)['answer_correct'])

    def test_format_stays_strict(self):
        self.assertFalse(m.score('Final answer: 23', '23')['format_ok'])
        self.assertTrue(m.score('</think><answer>23</answer>', '23')['format_ok'])


if __name__ == '__main__':
    unittest.main()
