from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from modelingmoe import BGEM3Model
import torch
import faiss
import numpy as np
import os
from tqdm import tqdm
import json

# ✅ GPU 지정
device = "cuda:5" if torch.cuda.is_available() else "cpu"
print(device)

# ✅ 모델 경로 및 이름
model_path = "/home/sunga/Desktop/BGE-M3-CLP-MoE-main/outputs/korean/"
model_name = "bge-m3-moe-korean"

# ✅ 토크나이저 및 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = BGEM3Model(
    model_name=model_path,
    colbert_dim=1024,
    moe="intermediate",
    only_train="intermediate",
    num_experts=2,
    num_experts_per_tok=1
).to(device)
model.eval()

langs = ["korean"]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        dense_vec, _, _ = model.encode(inputs)
        return dense_vec.cpu().numpy().flatten()

embedding_dir = os.path.join("/home/sunga/Desktop/Code/runs/bge-m3-moe", "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

for lang in langs:
    print(f"\n🔹 Processing language: {lang}...")

    corpus = load_dataset("castorini/mr-tydi-corpus", lang, split="train")
    queries = load_dataset("castorini/mr-tydi", lang, split="test")

    docid_to_text = {}
    for doc in corpus:
        text = doc.get("title", "") + " " + doc.get("text", "")
        if text.strip():
            docid_to_text[doc["docid"]] = text

    # ✅ 순서를 보존한 doc_ids/doc_texts
    doc_ids = list(docid_to_text.keys())
    doc_texts = list(docid_to_text.values())

    if len(doc_texts) == 0:
        print(f"❌ No documents found for {lang}, skipping.")
        continue

    doc_embed_file = os.path.join(embedding_dir, f"doc_embeddings.{lang}.{model_name}.npy")
    if os.path.exists(doc_embed_file):
        print(f" - Loading cached document embeddings from {doc_embed_file}")
        doc_embeddings = np.load(doc_embed_file)
    else:
        print(f" - Encoding {len(doc_texts)} documents...")
        doc_embeddings = np.array([get_embedding(text) for text in tqdm(doc_texts)])
        # np.save(doc_embed_file, doc_embeddings)
    faiss.normalize_L2(doc_embeddings)

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    query_ids = [q["query_id"] for q in queries]
    query_texts = [q["query"] for q in queries]

    query_embed_file = os.path.join(embedding_dir, f"query_embeddings.{lang}.{model_name}.npy")
    if os.path.exists(query_embed_file):
        print(f" - Loading cached query embeddings from {query_embed_file}")
        query_embeddings = np.load(query_embed_file)
    else:
        print(f" - Encoding and searching {len(query_texts)} queries...")
        # ✅ Query prefix 추가
        query_embeddings = np.array([get_embedding("Query: " + q) for q in tqdm(query_texts)])
        # np.save(query_embed_file, query_embeddings)
    faiss.normalize_L2(query_embeddings)

    D, I = index.search(query_embeddings, 10)

    output_dir = "/home/sunga/Desktop/Code/runs/bge-m3-moe"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"run.mrtydi.{lang}.{model_name}.trec")
    print(f" - Writing TREC results to {output_file}...")
    with open(output_file, "w") as fout:
        for idx, retrieved in enumerate(I):
            qid = str(query_ids[idx])
            for rank, doc_idx in enumerate(retrieved):
                raw_docid = doc_ids[doc_idx]
                clean_docid = raw_docid.split('#')[0]
                fout.write(f"{qid} Q0 {clean_docid} {rank+1} {D[idx][rank]:.4f} {model_name}\n")

    jsonl_file = os.path.join(output_dir, f"run.mrtydi.{lang}.{model_name}.jsonl")
    print(f" - Writing JSONL results to {jsonl_file}...")
    with open(jsonl_file, "w", encoding="utf-8") as fgen:
        for idx, retrieved in enumerate(I):
            qid = str(query_ids[idx])
            query = query_texts[idx]
            contexts = []
            for rank, doc_idx in enumerate(retrieved):
                docid = doc_ids[doc_idx]
                doc_text = doc_texts[doc_idx]
                score = float(D[idx][rank])
                contexts.append({
                    "docid": docid,
                    "rank": rank + 1,
                    "score": round(score, 4),
                    "text": doc_text
                })

            json_obj = {
                "query_id": qid,
                "query": query,
                "contexts": contexts
            }
            fgen.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("\n✅ All Mr.TyDi languages processed with BGE-M3-MoE.")


# nohup python bge-m3-moe.py > bge-m3-moe.out 2>&1 &