"""Transformers采样接口：原始token、停止位置和固定old分数。"""
from contextlib import nullcontext
import torch
from pipeline import grpo


def extract_completion(prefix, generated, end, decode):
    if not prefix or not generated or not 0 <= end <= len(generated):
        raise ValueError('采样长度无效')
    suffix=generated[:end] if end else generated
    return dict(text=decode(suffix),truncated=end==0,
                ids=prefix+suffix,mask=[0]*len(prefix)+[1]*len(suffix))


class TransformersBackend:
    def __init__(self,policy,tokenizer,config):
        self.policy,self.tokenizer,self.config=policy,tokenizer,config
        self.device=next(policy.parameters()).device

    def autocast(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else nullcontext()

    @torch.no_grad()
    def sample(self,prompts,max_new_tokens,stop_strings=()):
        from transformers import StoppingCriteria,StoppingCriteriaList,StopStringCriteria
        tokenizer=self.tokenizer
        device=self.device
        records=[]
        was_training=self.policy.training
        self.policy.eval()
        try:
            for start in range(0,len(prompts),self.config['generation_batch']):
                prefixes=[tokenizer.encode(p,add_special_tokens=False) for p in prompts[start:start+self.config['generation_batch']]]
                if any(not p for p in prefixes):raise ValueError('空prompt')
                width=max(map(len,prefixes))
                if width+max_new_tokens>self.config['max_length']:raise ValueError('生成预算超出上下文上限')
                class CaptureEnd(StoppingCriteria):
                    def __init__(self):
                        self.ends=torch.zeros(len(prefixes),device=device,dtype=torch.long)
                        self.stop=StopStringCriteria(tokenizer,list(stop_strings)) if stop_strings else None
                    def __call__(self,input_ids,scores,**kwargs):
                        done=input_ids[:,-1]==tokenizer.eos_token_id
                        if self.stop is not None:done=done|self.stop(input_ids,scores)
                        self.ends[done & (self.ends==0)]=input_ids.shape[1]-width
                        return done
                stop=CaptureEnd()
                ids=[[tokenizer.eos_token_id]*(width-len(p))+p for p in prefixes]
                attention=[[0]*(width-len(p))+[1]*len(p) for p in prefixes]
                with self.autocast():
                    output=self.policy.generate(input_ids=torch.tensor(ids,device=device),
                        attention_mask=torch.tensor(attention,device=device),do_sample=True,
                        temperature=1.,top_p=1.,top_k=0,typical_p=1.,min_p=None,
                        repetition_penalty=1.,no_repeat_ngram_size=0,min_new_tokens=0,
                        max_new_tokens=max_new_tokens,eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,forced_eos_token_id=None,
                        use_cache=True,stopping_criteria=StoppingCriteriaList([stop]))
                for prefix,tokens,end in zip(prefixes,output[:,width:].cpu().tolist(),stop.ends.cpu().tolist()):
                    records.append(extract_completion(prefix,tokens,end,
                        lambda x:tokenizer.decode(x,skip_special_tokens=True)))
                print('GENERATED',len(records),len(prompts),flush=True)
        finally:
            self.policy.train(was_training)
        return records

    def generate(self,prompts):
        # Proposer只接受Problem/Answer协议，训练适配性由audit阶段实测。
        return self.sample(prompts,self.config['problem_max_new_tokens'])

    @torch.no_grad()
    def solve(self,requests,group_size):
        prompts=[r['prompt'] for r in requests for _ in range(group_size)]
        samples=self.sample(prompts,self.config['solve_max_new_tokens'],['</answer>'])
        was_training=self.policy.training
        self.policy.eval()
        try:
            for index,sample in enumerate(samples):
                request=requests[index//group_size]
                sample.update(question_id=request['question_id'],candidate_index=index%group_size,
                              response=sample.pop('text'))
                ids=torch.tensor([sample['ids']],device=self.device)
                mask=torch.tensor([sample['mask']],device=self.device)
                with self.autocast():
                    logits=self.policy(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
                old,_=grpo.response_token_log_probs(logits,ids,mask)
                sample['old']=old[0].cpu().tolist()
        finally:
            self.policy.train(was_training)
        return samples
