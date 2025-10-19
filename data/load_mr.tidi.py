from datasets import load_dataset
import json
import os

def save_mrtydi_to_jsonl(lang_code, save_dir):
    try:
        dataset = load_dataset("castorini/mr-tydi", lang_code, split="train")
    except Exception as e:
        print(f"[{lang_code}] 로딩 실패: {e}")
        return

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{lang_code}_train_data.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for item in dataset:
            query = item["query"]
            pos_list = [p["text"] for p in item["positive_passages"]]
            neg_list = [p["text"] for p in item["negative_passages"]]

            sample = {
                "query": query,
                "pos": pos_list,
                "neg": neg_list
            }

            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    print(f"[{lang_code}] 저장 완료: {save_path}")

# 전체 언어 실행
LANGS = ["finnish", "indonesian", "korean", "japanese","english", "swahili", "thai","telugu","bengali", "arabic","russian"]

for lang in LANGS:
    save_mrtydi_to_jsonl(lang, save_dir="/home/sunga/Desktop/BGE-M3-CLP-MoE-main/data/")
