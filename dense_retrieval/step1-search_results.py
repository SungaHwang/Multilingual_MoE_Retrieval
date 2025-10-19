"""
python step1-search_results.py \
--encoder /home/sunga/Desktop/BGE-M3-CLP-MoE-main/outputs/all_final2-3-3 \
--languages arabic \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--threads 16 \
--hits 10 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False \
--moe intermediate


nohup python step1-search_results.py \
--encoder /home/sunga/Desktop/BGE-M3-CLP-MoE-main/outputs/all_final4-3-2 \
--languages thai \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--threads 16 \
--hits 10 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False \
--moe intermediate \
> thai.log 2>&1 &
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import torch
import datasets
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser, is_torch_npu_available
from _searcher import FaissSearcher, AutoQueryEncoder
from pyserini.output_writer import get_output_writer, OutputFormat

@dataclass
class ModelArgs:
    encoder: str = field(default="BAAI/bge-m3")
    add_instruction: bool = field(default=False)
    query_instruction_for_retrieval: str = field(default=None)
    pooling_method: str = field(default='cls')
    normalize_embeddings: bool = field(default=True)
    moe: str = field(default=None)

@dataclass
class EvalArgs:
    languages: str = field(default="en")
    index_save_dir: str = field(default='./corpus-index')
    result_save_dir: str = field(default='./search_results')
    threads: int = field(default=1)
    hits: int = field(default=1000)
    overwrite: bool = field(default=False)

def get_query_encoder(model_args: ModelArgs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_torch_npu_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")
    return AutoQueryEncoder(
        encoder_dir=model_args.encoder,
        device=device,
        pooling=model_args.pooling_method,
        l2_norm=model_args.normalize_embeddings,
        moe=model_args.moe
    )

def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    available_languages = ['bengali','arabic', 'swahili', 'en', 'es', 'finnish', 'hi', 'it', 'japanese', 'korean', 'pt', 'russian', 'thai', 'zh', 'fa', 'telugu', 'indonesian']
    for lang in languages:
        if lang not in available_languages:
            raise ValueError(f"Language `{lang}` is not supported. Available: {available_languages}")
    return languages

def get_queries_and_qids(lang: str, split: str='test', add_instruction: bool=False, query_instruction_for_retrieval: str=None):
    dataset = datasets.load_dataset('castorini/mr-tydi', lang, split=split, trust_remote_code=True)
    queries, qids = [], []
    for i in range(len(dataset)):
        qids.append(str(dataset[i]['query_id']))
        queries.append(str(dataset[i]['query']))
    if add_instruction and query_instruction_for_retrieval is not None:
        queries = [f"{query_instruction_for_retrieval}{query}" for query in queries]
    return queries, qids

def get_docid2text(lang):
    corpus = datasets.load_dataset("castorini/mr-tydi-corpus", lang, split="train", trust_remote_code=True)
    return {str(doc["docid"]): doc["text"] for doc in corpus}  # 🔥 string key로 강제 변환

def save_result(search_results: dict, result_save_path: str, qids: list, max_hits: int, queries: list, docid2text: dict):
    from pyserini.output_writer import get_output_writer, OutputFormat

    # ✅ TREC 저장은 그대로
    output_writer = get_output_writer(result_save_path, OutputFormat(OutputFormat.TREC.value), 'w',
                                      max_hits=max_hits, tag='Faiss', topics=qids,
                                      use_max_passage=False,
                                      max_passage_delimiter='#',
                                      max_passage_hits=1000)
    with output_writer:
        for topic, hits in search_results.items():
            output_writer.write(topic, hits)

    # ✅ JSONL 저장 (DenseSearchResult 객체 대응)
    jsonl_path = result_save_path.replace(".txt", ".jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, (qid, hits) in enumerate(search_results.items()):
            query = queries[i]
            contexts = []

            for rank, item in enumerate(hits):
                # item은 DenseSearchResult 객체라고 가정
                try:
                    docid = item.docid
                    score = item.score
                except AttributeError:
                    raise ValueError(f"[❌] Unexpected result format: {item}")

                docid_base = docid.split("#")[0]
                doc_text = docid2text.get(docid_base) or docid2text.get(docid) or ""

                contexts.append({
                    "docid": docid,
                    "rank": rank + 1,
                    "score": round(float(score), 4),
                    "text": doc_text
                })

            json_obj = {
                "query_id": qid,
                "query": query,
                "contexts": contexts,
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")



def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    languages = check_languages(eval_args.languages)

    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]

    query_encoder = get_query_encoder(model_args)
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)

    print("==================================================")
    print("Start generating search results with model:", model_args.encoder)
    print('Generate search results for languages: ', languages)

    for lang in languages:
        print("**************************************************")
        print(f"Start searching results for {lang} ...")

        result_save_path = os.path.join(eval_args.result_save_dir, os.path.basename(encoder), f"{lang}.txt")
        if not os.path.exists(os.path.dirname(result_save_path)):
            os.makedirs(os.path.dirname(result_save_path))
        if os.path.exists(result_save_path) and not eval_args.overwrite:
            print(f'Search results for {lang} already exist. Skipping...')
            continue

        index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder), lang)
        if not os.path.exists(index_save_dir):
            raise FileNotFoundError(f"{index_save_dir} not found")

        queries, qids = get_queries_and_qids(lang=lang, split='test',
                                             add_instruction=model_args.add_instruction,
                                             query_instruction_for_retrieval=model_args.query_instruction_for_retrieval)
        docid2text = get_docid2text(lang)

        searcher = FaissSearcher(index_dir=index_save_dir, query_encoder=query_encoder)
        search_results = searcher.batch_search(queries=queries, q_ids=qids,
                                               k=eval_args.hits, threads=eval_args.threads)

        # ✅ 그대로 넘긴다 (dict 형태 유지!)
        save_result(search_results, result_save_path, qids, eval_args.hits, queries, docid2text)

    print("==================================================")
    print("Finished generating search results with model:")
    pprint(model_args.encoder)

if __name__ == "__main__":
    main()
