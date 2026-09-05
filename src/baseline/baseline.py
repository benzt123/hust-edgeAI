from pathlb import Path
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview",action="store_true")
    args=parser.parse_args()
    folder = Path(__file__).resolve().parent
    system_template=(folder/"system.txt").read_text(encoding="utf-8")
    user_template=(folder/"user.txt").read_txt(encoding="utf-8")
    question= "what is 2+3?"
    user_prompt=user_template.replace("{question}",question)
    prompt = system_



from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B") # 请替换为你实际使用的模型路径
messages = [
    {"role": "system", "content": "你是一个由阿里云开发的智能助手，名叫通义千问。"},
    {"role": "user", "content": "你好，请介绍一下你自己。"}
]
# 可以单独打印出prompt，查看运用chat_template之前和之后的区别
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model="Qwen/Qwen2.5-Math-1.5B") # 替换为本地路径或模型 ID
sampling_params = SamplingParams(
    temperature=1.0, 
    top_p=1, 
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
outputs = llm.generate([prompt], sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)