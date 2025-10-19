# ✅ step2-eval_dense_mrtydi.py
import os
import json
import platform
import subprocess
import numpy as np
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from pyserini.util import download_evaluation_script

@dataclass
class EvalArgs:
    languages: str = field(default="en")
    encoder: str = field(default='BAAI/bge-m3')
    pooling_method: str = field(default='cls')
    normalize_embeddings: bool = field(default=True)
    search_result_save_dir: str = field(default='./search_results')
    qrels_dir: str = field(default='./qrels')
    metrics: str = field(default="ndcg@10", metadata={"nargs": "++"})
    eval_result_save_dir: str = field(default='./eval_results')

def map_metric(metric):
    k = metric.split('@')[-1]
    if 'ndcg' in metric: return k, f'ndcg_cut.{k}'
    if 'recall' in metric: return k, f'recall.{k}'
    raise ValueError(f"Unknown metric {metric}")

def evaluate(script_path, qrels_path, result_path, metrics):
    results = {}
    for metric in metrics:
        k, mapped = map_metric(metric)
        cmd = ['java', '-jar', script_path, '-c', '-M', str(k), '-m', mapped, qrels_path, result_path]
        shell = platform.system() == "Windows"
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        out, err = proc.communicate()
        try:
            results[metric] = float(out.decode().split()[-1].strip())
        except:
            results[metric] = out.decode()
    return results

def main():
    parser = HfArgumentParser([EvalArgs])
    args = parser.parse_args_into_dataclasses()[0]

    script_path = download_evaluation_script('trec_eval')
    lang = args.languages
    encoder_name = os.path.basename(args.encoder.rstrip('/'))
    qrels_path = os.path.join(args.qrels_dir, f"mrtydi-{lang}-qrels.tsv")
    result_path = os.path.join(args.search_result_save_dir, encoder_name, f"{lang}.txt")
    results = evaluate(script_path, qrels_path, result_path, args.metrics)

    os.makedirs(args.eval_result_save_dir, exist_ok=True)
    save_path = os.path.join(args.eval_result_save_dir, f"{encoder_name}.json")
    json.dump({"results": results}, open(save_path, 'w'), indent=2)
    print(f"✅ Evaluation saved to {save_path}")

if __name__ == "__main__":
    main()
