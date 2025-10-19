from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from modelingmoe import BGEM3Model
import torch
import numpy as np
import os

# ✅ GPU 설정
device = "cuda:5" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ✅ 모델 경로
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

# ✅ 문서 로딩
corpus = load_dataset("castorini/mr-tydi-corpus", "korean", split="train")

# ✅ docid_to_text 생성
docid_to_text = {}
for doc in corpus:
    text = doc.get("title", "") + " " + doc.get("text", "")
    if text.strip():
        docid_to_text[doc["docid"]] = text

# ✅ 정답 쿼리와 문서
query = "한니발 바르카의 최종 계급은 무엇인가요?"
docid = "7207#0"

if docid not in docid_to_text:
    print(f"❌ 정답 문서 {docid} 가 corpus에 없습니다.")
    exit()

doc_text = docid_to_text[docid]

# ✅ 임베딩 함수
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        dense_vec, _, _ = model.encode(inputs)
        return dense_vec.cpu().numpy().flatten()

# ✅ 유사도 계산
query_vec = get_embedding("Query: " + query)
doc_vec = get_embedding(doc_text)
cos_sim = np.dot(query_vec, doc_vec)

print(f"\n🧪 Cosine similarity between query and gold doc ({docid}): {cos_sim:.4f}")
