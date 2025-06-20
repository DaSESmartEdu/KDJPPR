import torch
import json
from tqdm import tqdm
import faiss
import numpy as np


def get_dict(dict_path):
    with open(dict_path, 'r') as f:
        return json.load(f)

def get_embeddings(emb_path):
    return torch.load(emb_path)

def normalize_embeddings(embeddings):
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    return embeddings / norms




def find_top_k_apis(
    API_embeddings: torch.Tensor,
    skill_embeddings: torch.Tensor,
    API_dict: dict,
    skill_dict: dict,
    top_k: int,
    use_gpu: bool = True,
    save_to_file: bool = False,
    output_file: str = "output.json",
    use_cosine_similarity: bool = True  
) -> dict:
    if use_cosine_similarity:
        API_embeddings = normalize_embeddings(API_embeddings)
        skill_embeddings = normalize_embeddings(skill_embeddings)
    
    API_embeddings_np = API_embeddings.numpy().astype(np.float32)
    skill_embeddings_np = skill_embeddings.numpy().astype(np.float32)

    index = faiss.IndexFlatL2(API_embeddings_np.shape[1])

    if use_gpu:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        index_to_use = gpu_index
    else:
        index_to_use = index
    
    index_to_use.add(API_embeddings_np)
    D, I = index_to_use.search(skill_embeddings_np, top_k)
       
    skill_to_apis = {}
    for skill_idx, api_indices in enumerate(I):
        skill_name = skill_dict[skill_idx]  
        api_names = [API_dict[api_idx] for api_idx in api_indices]  
        skill_to_apis[skill_name] = api_names
    
    return skill_to_apis


top_k_list = 30

results = find_top_k_apis(
        API_embeddings=API_embeddings,
        skill_embeddings=skill_embeddings,
        API_dict=id2API,
        skill_dict=id2skill,
        top_k=top_k_list,
        use_gpu=True,
        use_cosine_similarity=True
    )

output_file = "output.json"
with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)