import json
import re
from tqdm import tqdm

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

all_skills = []
with open('raw_data.json') as f:
    data = json.load(f)
for item in tqdm(data):
    answer= item['answer'] 
    skills = get_skills(answer) 
    if len(skills) != 0:
        all_skills.append(skills)
        
skill_dict = {}
for idx, skill in enumerate(new_skills):
    skill_dict[skill] = idx

with open('skills.json', 'w') as f:
    json.dump(skill_dict, f)
    
    
    
    
titles = []
with open('raw_data.json') as f:
    data = json.load(f)
for item in tqdm(data):
    question = item['question'] 
    title = get_title(question) 
    titles.append(title.strip())
set_titles = list(set(titles))
general_titles = {}
for idx, title in enumerate(set_titles):
    general_titles[title] = idx
with open('./general_titles.json', 'w') as f:
    json.dump(general_titles, f)