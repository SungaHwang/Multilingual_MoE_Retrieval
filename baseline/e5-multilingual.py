from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os
from tqdm import tqdm
import json

device = "cuda:6" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

langs = ["korean", "swahili", 
         "telugu", "thai", "arabic", "bengali", "finnish"]
# "indonesian", 

# ✅ Mean pooling function for E5 embeddings
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ✅ E5 전용 임베딩 함수 (프롬프트 + mean pooling 포함)
def get_embedding(text, prefix):
    text = prefix + text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        return emb.cpu().numpy().flatten()

embedding_dir = os.path.join("/home/sunga/Desktop/Code/runs/e5-multilingual", "embeddings")
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

    doc_ids = sorted(docid_to_text.keys())
    doc_texts = [docid_to_text[docid] for docid in doc_ids]

    if len(doc_texts) == 0:
        print(f"❌ No documents found for {lang}, skipping.")
        continue

    # ✅ 문서 임베딩 캐시 확인
    doc_embed_file = os.path.join(embedding_dir, f"doc_embeddings.{lang}.{model_name.replace('/', '_')}.npy")
    if os.path.exists(doc_embed_file):
        print(f" - Loading cached document embeddings from {doc_embed_file}")
        doc_embeddings = np.load(doc_embed_file)
    else:
        print(f" - Encoding {len(doc_texts)} documents...")
        doc_embeddings = np.array([get_embedding(text, prefix="passage: ") for text in tqdm(doc_texts)])
        #np.save(doc_embed_file, doc_embeddings)
    faiss.normalize_L2(doc_embeddings)

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    query_ids = [q["query_id"] for q in queries]
    query_texts = [q["query"] for q in queries]

    # ✅ 쿼리 임베딩 캐시 확인
    query_embed_file = os.path.join(embedding_dir, f"query_embeddings.{lang}.{model_name.replace('/', '_')}.npy")
    if os.path.exists(query_embed_file):
        print(f" - Loading cached query embeddings from {query_embed_file}")
        query_embeddings = np.load(query_embed_file)
    else:
        print(f" - Encoding {len(query_texts)} queries...")
        query_embeddings = np.array([get_embedding(q, prefix="query: ") for q in tqdm(query_texts)])
        #np.save(query_embed_file, query_embeddings)
    faiss.normalize_L2(query_embeddings)

    D, I = index.search(query_embeddings, 10)

    output_dir = "/home/sunga/Desktop/Code/runs/e5-multilingual"
    os.makedirs(output_dir, exist_ok=True)

    # TREC 파일 저장
    output_file = os.path.join(output_dir, f"run.mrtydi.{lang}.e5.trec")
    print(f" - Writing TREC results to {output_file}...")
    with open(output_file, "w") as fout:
        for idx, retrieved in enumerate(I):
            qid = str(query_ids[idx])
            for rank, doc_idx in enumerate(retrieved):
                fout.write(f"{qid} Q0 {doc_ids[doc_idx]} {rank+1} {D[idx][rank]:.4f} E5-Base-v2\n")

    # JSONL 파일 저장
    jsonl_file = os.path.join(output_dir, f"run.mrtydi.{lang}.e5.jsonl")
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

print("\n✅ All Mr.TyDi languages processed with E5-Base-v2 (prompt-corrected, mean-pooled, TREC and JSONL outputs).")
