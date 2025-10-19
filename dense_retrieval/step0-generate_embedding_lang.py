"""
python step0-generate_embedding.py \
--encoder /home/sunga/Desktop/BGE-M3-CLP-MoE-main/outputs/all_final2-3-1 \
--languages swahili \
--index_save_dir ./corpus-index \
--max_passage_length 512 \
--batch_size 4 \
--fp16 \
--pooling_method cls \
--normalize_embeddings True \
--moe intermediate

nohup python step0-generate_embedding.py \
--encoder /home/sunga/Desktop/BGE-M3-CLP-MoE-main/outputs/all_final2-3-1 \
--languages swahili \
--index_save_dir ./corpus-index \
--max_passage_length 512 \
--batch_size 4 \
--fp16 \
--pooling_method cls \
--normalize_embeddings True \
--moe intermediate \
> logs/swahili3-3-1.log 2>&1 &

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import faiss
import datasets
import numpy as np
from tqdm import tqdm
from flag_models import FlagModel
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ModelArgs:
    encoder: str = field(default="BAAI/bge-m3")
    fp16: bool = field(default=True)
    pooling_method: str = field(default='cls')
    normalize_embeddings: bool = field(default=True)
    moe: str = field(default=None)

@dataclass
class EvalArgs:
    languages: str = field(default="en")
    index_save_dir: str = field(default='./corpus-index')
    max_passage_length: int = field(default=512)
    batch_size: int = field(default=256)
    overwrite: bool = field(default=False)

# 언어별 특수 토큰
SPECIAL_LANG_TOKENS = {
    'arabic': '[ar]', 'bengali': '[bn]', 'english': '[en]', 'finnish': '[fi]',
    'indonesian': '[id]', 'japanese': '[ja]', 'korean': '[ko]', 'russian': '[ru]',
    'telugu': '[te]', 'thai': '[th]', 'swahili': '[sw]'
}

# 언어별 어순 매핑
LANG_ORDER_MAP = {
    'korean': '[SOV]',
    'arabic': '[VSO]',
    'bengali': '[SOV]',
    'finnish': '[SVO]',
    'indonesian': '[SVO]',
    'swahili': '[SVO]',
    'telugu': '[SOV]',
    'thai': '[SVO]'
}

# 다양성 계산
def estimate_diversity(text):
    tokens = text.split()
    unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
    if unique_ratio > 0.6:
        return '[D_H]'
    elif unique_ratio > 0.3:
        return '[D_M]'
    else:
        return '[D_L]'

# 길이 계산
def estimate_length(text):
    length = len(text.split())
    if length > 100:
        return '[L_H]'
    elif length > 50:
        return '[L_M]'
    else:
        return '[L_L]'

# 입력 텍스트에 피처 토큰을 붙이기
def prepend_features(texts, lang):
    lang_token = SPECIAL_LANG_TOKENS.get(lang, '[unk]')
    order_token = LANG_ORDER_MAP.get(lang, '[SVO]')  # 기본 어순: SVO
    new_texts = []
    for t in texts:
        diversity_token = estimate_diversity(t)
        length_token = estimate_length(t)
        new_texts.append(f"{lang_token} {order_token} {diversity_token} {length_token} {t}")
    return new_texts

def get_model(model_args: ModelArgs):
    return FlagModel(
        model_args.encoder,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16,
        moe=model_args.moe
    )

def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    available = list(SPECIAL_LANG_TOKENS.keys())
    for lang in languages:
        if lang not in available:
            raise ValueError(f"Language `{lang}` is not supported. Available: {available}")
    return languages

def load_corpus(lang: str):
    corpus = datasets.load_dataset('castorini/mr-tydi-corpus', lang, split='train', trust_remote_code=True)
    corpus_list = []
    for e in tqdm(corpus, desc=f"Loading MR-TyDi corpus ({lang})"):
        corpus_list.append({'id': e['docid'], 'content': e['text']})
    return datasets.Dataset.from_list(corpus_list)

def generate_index(model: FlagModel, corpus: datasets.Dataset, lang: str, max_passage_length: int, batch_size: int):
    texts_with_features = prepend_features(corpus["content"], lang)
    corpus_embeddings = model.encode_corpus(texts_with_features, batch_size=batch_size, max_length=max_passage_length)
    dim = corpus_embeddings.shape[-1]
    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index, list(corpus["id"])

def save_result(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)

def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()

    languages = check_languages(eval_args.languages)
    model = get_model(model_args)

    encoder = model_args.encoder.rstrip('/')
    print("==================================================")
    print("Start generating embedding with model:", encoder)
    print("Languages:", languages)

    for lang in languages:
        print("**************************************************")
        index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder), lang)
        os.makedirs(index_save_dir, exist_ok=True)

        if os.path.exists(os.path.join(index_save_dir, 'index')) and not eval_args.overwrite:
            print(f"Embedding of {lang} already exists. Skipping...")
            continue

        print(f"Start generating embedding of {lang} ...")
        corpus = load_corpus(lang)
        index, docid = generate_index(
            model=model,
            corpus=corpus,
            lang=lang,
            max_passage_length=eval_args.max_passage_length,
            batch_size=eval_args.batch_size
        )
        save_result(index, docid, index_save_dir)

    print("==================================================")
    print("Finished generating embeddings with model:", encoder)

if __name__ == "__main__":
    main()
