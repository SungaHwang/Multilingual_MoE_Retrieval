from datasets import load_dataset
import os

langs = ["finnish", "indonesian", "japanese", "korean", "russian", 
         "swahili", "telugu", "thai", "arabic", "bengali", "english"]

for lang in langs:
    print(f"📄 Generating qrels for {lang}...")

    queries = load_dataset("castorini/mr-tydi", lang, split="test")

    os.makedirs("/home/sunga/Desktop/Code/qrels", exist_ok=True)
    output_path = f"/home/sunga/Desktop/Code/qrels/qrels.mrtydi.{lang}.txt"

    with open(output_path, "w") as fout:
        for q in queries:
            qid = str(q["query_id"])
            for pos in q["positive_passages"]:
                docid = str(pos["docid"])
                fout.write(f"{qid} 0 {docid} 1\n")  # ⬅ 공백 구분

    print(f"✅ Saved to {output_path}")
