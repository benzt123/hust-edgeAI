"""受控算术课程：整数精确答案，不调用模型生成真值。"""
import random


def generate(count, seed):
    rng = random.Random(seed)
    result = []
    for i in range(count):
        kind = i % 3
        if kind == 0:
            boxes, per_box = rng.randint(12, 45), rng.randint(15, 40)
            damaged = rng.randint(2, 10)
            sold = rng.randint(20, 80)
            question = (f'A shop receives {boxes} boxes containing {per_box} pencils each. '
                        f'Exactly {damaged} pencils in EACH box are damaged and discarded. '
                        f'The shop then sells {sold} of the undamaged pencils. '
                        'How many undamaged pencils remain?')
            answer = boxes * (per_box - damaged) - sold
            assert answer == boxes * per_box - boxes * damaged - sold and answer >= 0
            parameters = dict(boxes=boxes, per_box=per_box, damaged=damaged, sold=sold)
        elif kind == 1:
            price = 400 * rng.randint(2, 15)
            quantity = rng.randint(3, 12)
            discount = rng.choice([10, 20, 25, 30, 40])
            tax = rng.choice([5, 10, 15, 20])
            shipping = rng.randint(8, 40)
            question = (f'A store sells a chair for ${price} before discounts. A customer buys '
                        f'{quantity} chairs, each discounted by {discount} percent. A sales tax '
                        f'of {tax} percent applies to the discounted merchandise total. '
                        f'A single shipping fee of ${shipping} is added AFTER tax and is not taxed. '
                        'How many dollars does the customer pay in total?')
            subtotal = price * quantity * (100-discount) // 100
            assert subtotal * tax % 100 == 0
            answer = subtotal + subtotal * tax // 100 + shipping
            assert answer * 10000 == price * quantity * (100-discount) * (100+tax) + shipping * 10000
            parameters = dict(price=price, quantity=quantity, discount=discount, tax=tax, shipping=shipping)
        else:
            fast, slow = rng.randint(15, 35), rng.randint(4, 12)
            together, alone = rng.randint(12, 40), rng.randint(5, 18)
            rejected = rng.randint(10, 50)
            question = (f'Machine A makes {fast} parts per minute and machine B makes {slow} '
                        f'parts per minute. Both run for {together} minutes. Then B stops and A '
                        f'runs alone for {alone} more minutes. Of all parts produced, exactly '
                        f'{rejected} are defective and discarded. How many usable parts remain?')
            answer = (fast + slow) * together + fast * alone - rejected
            assert answer == fast * (together + alone) + slow * together - rejected and answer > 0
            parameters = dict(fast=fast, slow=slow, together=together, alone=alone, rejected=rejected)
        result.append(dict(text=f'Problem: {question}\nAnswer: {answer}', truncated=False,
                           provenance=dict(generator='verified_arithmetic_v1', family=kind,
                                           parameters=parameters, exact_answer=answer)))
    return result
