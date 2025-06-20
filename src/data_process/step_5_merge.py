import json
from tqdm import tqdm
import re

def get_dict(dict_path):
    with open(dict_path, "r") as f:
        data = json.load(f)
    return data
  
  
distill_data_path = r'raw_distill_data.json'
distill_data = get_dict(distill_data_path)

skill_API_replace_path = r'skill_API_replace.json'
skill_API_replace = get_dict(skill_API_replace_path)

title_standardiztion_path = r'title_standardiztion_.json'
title_standardiztion = get_dict(title_standardiztion_path)

def get_skills(answer):
    skills_pattern = re.compile(r'The skills are (.*?)\.', re.DOTALL)
    match = skills_pattern.search(answer)
    if match:
        skills = [skill.strip() for skill in match.group(1).split(',')]
        skills = [skill.strip() for skill in skills]
        return skills


def get_title(question):
    job_title_pattern = re.compile(r'What skills are required for the role of (.+?)\? Here is the job description')
    match = job_title_pattern.search(question)
    if match:
        job_title = match.group(1)
        return job_title

def remove_similar(lst):
    unique_list = []
    for item in lst:
        if not any(item.lower() == existing.lower() for existing in unique_list):
            unique_list.append(item)
    return unique_list



new_dis_data = []

for item in tqdm(distill_data):
    question = item['question']
    answer = item['answer']

    skills = get_skills(answer)
    job_title = get_title(question)

    matched_title = title_match[job_title]
    APIs = []
    for skill in skills:
        if len(skill) > 1:
            matched_API = skill_API_match[skill]
            macthed_API = [API for API in matched_API if len(API) > 2]
            APIs.extend(matched_API)
    APIs = list(set(APIs))
    APIs = remove_similar(APIs)
    if len(APIs) > 0:
        APIs = ", ".join(APIs)
        
        question = question.replace('What skills are required for', 'What APIs are required for')
        question = question.replace(f'for the role of {job_title}?', f'for the role of {matched_title}?')
        
        new_answer = "The APIs are" + " " + APIs + "."
        new_item = {
            'question': question,
            'answer': new_answer
        }
        new_dis_data.append(new_item)


with open("distill_data.json", "w", encoding="utf-8") as f:
    json.dump(new_dis_data, f, ensure_ascii=False, indent=4)