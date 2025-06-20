import argparse
import json
import os
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser(description="Evaluate model on project participation prediction")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the prediction results")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
parser.add_argument("--device", type=str, default="0", help="Specify the GPU device ID (e.g., '0', '1', '2')")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device


def extract_career_history(text):
    match = re.search(r'career history are \[(.*?)\]\.', text)
    return match.group(1).split(", ") if match else []

def extract_project_experience(text):
    match = re.search(r'project experience are (\[\[.*?\]\])', text)
    return match.group(1) if match else []


def extract_target_project(question):
    match = re.search(r'target project:\s*"([^"]+)"', question)
    return match.group(1) if match else []



class ProjectDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.system_prompt = (
            "You are a helpful assistant.. Given a user’s Career History and Open Source Project Experience "
            "(represented by corresponding APIs in the open-source project), you must determine whether the user is likely to participate in the target project"
        )

    def get_prompt(self, career_history, project, target_project):
        prompt = (
            f'Career History: {career_history}\n'
            f'Open Source Project Experience: {project}\n'
            f'Analyze user\'s career history and open source project experience to determine whether user is likely to participate in the target Project: "{target_project}"\n\n'
            f'You MUST output a valid JSON object in the EXACT format below:\n\n'
            f'```json\n'
            f'{{"result": "Yes" or "No"}}\n'
            f'```\n\n'
            f'- STRICT FORMAT REQUIREMENT: Do NOT include explanations, extra text, comments, or any other formatting—ONLY return the JSON object.\n'
        )
        return prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["Input"]

        career_history = extract_career_history(input_text)
        project_experience = extract_project_experience(input_text)
        target_project = extract_target_project(input_text)
        
        if career_history == [] or project_experience == [] or target_project == []:
            print(input_text)
        prompt = self.get_prompt(career_history, project_experience, target_project)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        batch_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return {
            "batch_text": batch_text,
            "label": item["Label"],
            "original_item": item
        }


def collate_fn(batch):
    batch_texts = [item["batch_text"] for item in batch]
    labels = [item["label"] for item in batch]
    original_items = [item["original_item"] for item in batch]
    return batch_texts, labels, original_items


class EvaluateData:
    def __init__(self, tokenizer, model, sampling_params):
        self.tokenizer = tokenizer
        self.model = model
        self.sampling_params = sampling_params
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)


    def generate(self, batch_texts):
        outputs = self.model.generate(batch_texts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = LLM(model=args.model_path)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=2048)
evaluator = EvaluateData(tokenizer, model, sampling_params)


dataset = ProjectDataset(args.data_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


results = []
for batch_texts, labels, original_items in tqdm(dataloader):
    responses = evaluator.generate(batch_texts)
    for response, item in zip(responses, original_items):
        item["Prediction"] = response
        results.append(item)

model_name = args.model_path.split("/")[-1]
data_name = args.data_path.split("/")[-1].split(".")[0]
output_path = os.path.join(args.output_path, f"results_{data_name}_{model_name}.json")

print(f"Saving results to {output_path}")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
