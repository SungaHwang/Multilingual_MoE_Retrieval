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

# ✅ LaBSE 모델 로드
model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

langs = ["telugu", "thai", "arabic", "bengali", "finnish"]
# "indonesian", "korean", "swahili", 

# ✅ Mean pooling 함수 (LaBSE는 CLS 대신 mean pooling 권장)
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ✅ 임베딩 추출
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
    return embedding

output_dir = "/home/sunga/Desktop/Code/runs/labse"
embedding_dir = os.path.join(output_dir, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

for lang in langs:
    print(f"\n🔹 Processing language: {lang}...")

    # Load datasets
    corpus = load_dataset("castorini/mr-tydi-corpus", lang, split="train")
    queries = load_dataset("castorini/mr-tydi", lang, split="test")

    # Document Processing
    docid_to_text = {}
    for doc in corpus:
        text = doc.get("title", "") + " " + doc.get("text", "")
        if text.strip():
            docid_to_text[doc["docid"]] = text

    doc_ids = sorted(docid_to_text.keys())
    doc_texts = [docid_to_text[docid] for docid in doc_ids]

    if not doc_texts:
        print(f"❌ No documents found for {lang}, skipping.")
        continue

    # Document Embeddings
    doc_embed_file = os.path.join(embedding_dir, f"doc_embeddings.{lang}.{model_name.replace('/', '_')}.npy")
    if os.path.exists(doc_embed_file):
        print(f" - Loading cached document embeddings from {doc_embed_file}")
        doc_embeddings = np.load(doc_embed_file)
    else:
        print(f" - Encoding {len(doc_texts)} documents...")
        doc_embeddings = np.array([get_embedding(text) for text in tqdm(doc_texts)])
        #np.save(doc_embed_file, doc_embeddings)

    # Indexing (Inner Product)
    faiss.normalize_L2(doc_embeddings)
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    # Query Processing
    query_ids = [q["query_id"] for q in queries]
    query_texts = [q["query"] for q in queries]

    # Query Embeddings
    query_embed_file = os.path.join(embedding_dir, f"query_embeddings.{lang}.{model_name.replace('/', '_')}.npy")
    if os.path.exists(query_embed_file):
        print(f" - Loading cached query embeddings from {query_embed_file}")
        query_embeddings = np.load(query_embed_file)
    else:
        print(f" - Encoding {len(query_texts)} queries...")
        query_embeddings = np.array([get_embedding(q) for q in tqdm(query_texts)])
        #np.save(query_embed_file, query_embeddings)

    faiss.normalize_L2(query_embeddings)
    D, I = index.search(query_embeddings, 10)

    # Save TREC Format
    trec_file = os.path.join(output_dir, f"run.mrtydi.{lang}.labse.trec")
    with open(trec_file, "w") as fout:
        for idx, retrieved in enumerate(I):
            qid = str(query_ids[idx])
            for rank, doc_idx in enumerate(retrieved):
                fout.write(f"{qid} Q0 {doc_ids[doc_idx]} {rank + 1} {D[idx][rank]:.4f} LaBSE\n")

    # Save JSONL Format
    jsonl_file = os.path.join(output_dir, f"run.mrtydi.{lang}.labse.jsonl")
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

print("\n✅ All Mr.TyDi languages processed with LaBSE.")
