from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class LLM4JobTitleMatcher:
    def __init__(self, tokenizer, model, sampling_params):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.model = model
        self.sampling_params = sampling_params
        self.system_prompt = (
            "You are a helpful assistant. Given Job Title,"
            "you need to find the most relevant job title from a candidate set of standardized job titles."
        )
    
    def get_prompt(self, job_title, candidates):
        candidate_string = "\n".join(f"[{candidate}]" for candidate in candidates)
        prompt = (
            f'Job Title:\n "{job_title}"\n'
            f'Candidates:\n"{candidate_string}"\n'
            f'Please output the the most relevant job title from Candidates based on the given Job Title: "{job_title}"\n'
            f"Your response MUST be a valid JSON object in this exact format:\n"
            f'{{"s_title": "Candidate"}}\n'
            f"- Replace 'Candidate' with exactly ONE candidate from the candidates.\n"
            f"- Do NOT include any explanations, additional text, formatting or reasoning.\n"
        )
        return prompt
    
    def generate(self, job_titles, candidates_list):
        prompt_batch = [self.get_prompt(job_title, candidates) for job_title, candidates in zip(job_titles, candidates_list)]
        messages_batch = [
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            for prompt in prompt_batch
        ]
        batch_texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        outputs = self.model.generate(batch_texts, self.sampling_params)

        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)

        return responses

model_path = "Qwen2.5-14B model path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LLM(model=model_path)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
matcher = LLM_Prompt(tokenizer,model,sampling_params)


with open('job_title_candidates.json','r') as f:
    data = json.load(f)

batch_size = 128
job_titles = list(data.keys())
candidates_list = list(data.values())
all_results = []
for i in tqdm(range(0, len(job_titles), batch_size)):
    current_job_titles = job_titles[i:i + batch_size]
    current_candidates = candidates_list[i:i + batch_size]
    prompt_results = matcher.generate(current_job_titles, current_candidates)
    for job, candidates, result in zip(current_job_titles, current_candidates, prompt_results):
        result = {job:candidates,"standardized_title":result}
        all_results.append(result)
    #break
with open('standardized_titles.json', 'w') as json_file:
    json.dump(all_results, json_file, indent=4)
