"""检查配置约束并生成实际 prompt 示例；此脚本不会启动训练。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    c = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    assert c['model']['restart_from_base'] and 'Instruct' not in c['model']['id']
    assert c['prompt']['mode'] == 'user_r1_zero_only'
    assert not c['prompt']['apply_chat_template']
    text = (HERE/c['prompt']['file']).read_text(encoding='utf-8').rstrip()
    assert text.count('{question}') == 1
    assert text.endswith('Assistant: <think>')
    assert 'please delete this line' not in text
    assert c['dataset']['validation_generation_limit'] == c['dataset']['test_generation_limit'] == 0
    assert not c['selection']['test_used_for_selection']
    assert c['stages']['rsft']['input'] == 'sft.best_model'
    assert c['stages']['dpo']['input'] == 'rsft.best_model'
    g = c['stages']['grpo']
    assert g['input'] == 'dpo.best_model'
    assert g['effective_batch'] == g['logical_train_batch_size']*g['logical_gradient_accumulation_steps']
    assert (g['rollout_rounds'],g['rollout_questions'],g['group_size'],g['epochs_per_rollout']) == (200,32,8,2)
    assert g['learning_rate'] == 1e-5 and g['optimizer_betas'] == [.9,.95]
    assert g['advantage'] == 'reward_minus_group_mean' and not g['advantage_eps_used']
    assert c['generation']['max_new_tokens'] == 1024 and c['generation']['grpo_min_tokens'] == 4
    assert c['grader']['function'] == 'r1_zero_reward_fn' and c['grader']['fast'] is False
    example = text.replace('{question}',r'Solve for x: $x^2=2$. Give both real solutions.')
    (HERE/'prompts'/'preview.txt').write_text(example,encoding='utf-8')
    print('Configuration consistency OK; this does not validate a GPU training runtime.')
    print('GRPO planned rollouts:',200*32*8,'responses; optimizer updates:',200*(32*8//32)*2)
    print('Prompt preview:',HERE/'prompts'/'preview.txt')


if __name__ == '__main__':
    main()
