from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class LLM_Prompt:
    def __init__(self, tokenizer, model, sampling_params):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
        self.model = model
        self.sampling_params = sampling_params
        self.system_prompt = (
            "You are a helpful assistant. Given Skill, "
            "you need to find the most relevant APIs from a candidate set of APIs."
        )

    
    def get_prompt(self, skill, candidates):
        candidate_string = "\n".join(f"[{candidate}]" for candidate in candidates)
        prompt = (
            f'Skill: "{skill}"\n'
            f'Select the most relevant API(s) from the following Candidates based on the given skill:\n'
            f'Candidates:\n"{candidate_string}"\n'
            f'Please output the the most relevant API(s) from Candidates based on the given Skill.'
            f"Your response MUST be a valid JSON object in this exact format:\n"
            f'{{"result": ["Candidate1", "Candidate2", ...]}}\n'
            f"- Replace 'Candidate1', 'Candidate2', etc., with one or more valid candidates.\n"
            f"- Ensure the response is a valid JSON object with double quotes around keys and values.\n"
            f"- Do NOT include explanations, additional text, or any extra formatting or reasoning.\n"
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


with open('candidate_skill_API.json','r') as f:
    data = json.load(f)

batch_size = 128
skills = list(data.keys())
candidates_list = list(data.values())
all_results = []
for i in tqdm(range(1, len(skills), batch_size)):
    current_skills = skills[i:i + batch_size]
    current_candidates = candidates_list[i:i + batch_size]

    new_current_candidates = []
    for sublist in current_candidates:
        items = []
        for item in sublist:
            if len(item) > 1:
                items.append(item)
        new_current_candidates.append(items)
    current_candidates = new_current_candidates
    prompt_results = matcher.generate(current_skills, current_candidates)
    for skill, candidates, result in zip(current_skills, current_candidates, prompt_results):
        result = {skill:candidates,"APIs":result}
        all_results.append(result)

with open('replace_skill_APIs.json', 'w') as json_file:
    json.dump(all_results, json_file, indent=4)