import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings for given API/Job Title data using a transformer model.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to the JSON dictionary containing text items.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing (adjust based on GPU memory).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted embeddings.")
    return parser.parse_args()

class EmbeddingExtractor:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = 32
    
    def encode_texts(self, texts: list) -> torch.Tensor:
        embeddings = self.model.encode(texts, prompt_name="query")
        return embeddings

class MyDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def get_dict(dict_path):
    with open(dict_path, "r") as f:
        data = json.load(f)
    return data

def main():
    args = parse_args()
    
    
    data_dict = get_dict(args.dict_path)
    items = list(data_dict.keys())
    num_items = len(items)
    
    embeddings_tensor = torch.zeros((num_items, 1536), dtype=torch.float16)
    model_name = "gte-Qwen2-1.5B-instruct model path"
    extractor = EmbeddingExtractor(model_name)

    dataset = MyDataset(items)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        embeddings = extractor.encode_texts(batch)
        embeddings = torch.from_numpy(embeddings)
        for item, embedding in zip(batch, embeddings):
            item_id = data_dict[item]
            embeddings_tensor[item_id] = embedding
        
    
    torch.save(embeddings_tensor, args.output_path)

if __name__ == "__main__":
    main()
