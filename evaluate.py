import pytrec_eval

def load_qrels(qrels_path):
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            docid = docid.split("#")[0]
            qrels.setdefault(str(qid), {})[docid] = int(rel)
    return qrels


def load_run(run_path):
    run = {}
    with open(run_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score, _ = parts
            qid = str(qid)
            docid = docid.split("#")[0]
            run.setdefault(qid, {})[docid] = float(score)
    return run


qrels_path = 'qrels/qrels.mrtydi.telugu.txt'
#run_path = 'runs/bge-m3/run.mrtydi.arabic.bge-m3.trec'
#run_path = 'runs/bge-m3-moe/swahili.trec'
#run_path = 'runs/bge-m3-moe/bengali.trec'
#run_path = 'runs/bge-m3/run.mrtydi.swahili.bge-m3.trec'

#run_path = 'runs/mcontriever/run.mrtydi.finnish.mcontriever.trec'
#run_path = 'runs/e5-multilingual/run.mrtydi.finnish.e5.trec'
#run_path = 'runs/labse/run.mrtydi.finnish.labse.trec'
run_path = '/home/sunga/Desktop/bge-m3-moe/dense_retrieval/search_results/all_final3-3-3/telugu.txt'

qrels = load_qrels(qrels_path)
run = load_run(run_path)

# 디버깅: 쿼리 ID 겹치는지 확인
overlap_qids = set(qrels.keys()) & set(run.keys())
print(f"✅ Overlapping query IDs: {len(overlap_qids)}")

# 겹치는 QID 하나라도 있으면 docid도 확인해보기
if overlap_qids:
    sample_qid = next(iter(overlap_qids))
    print(f"🔍 Sample QID: {sample_qid}")
    print("   Qrels docids:", list(qrels[sample_qid].keys())[:5])
    print("   Run docids  :", list(run[sample_qid].keys())[:5])
else:
    print("❌ No overlapping query IDs between qrels and run.")
    exit()

# 평가 지표
metrics = {
    'recip_rank',
    'map',
    'ndcg_cut_1',
    'ndcg_cut_2',
    'ndcg_cut_3',
    'P_1',
    'P_5',
    'P_10',
    'Rprec',
    'recall_10',
    'bpref',
}

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
results = evaluator.evaluate(run)

# 평균값 계산
avg_metrics = {metric: 0.0 for metric in metrics}
for query_id in results:
    for metric in metrics:
        avg_metrics[metric] += results[query_id][metric]
for metric in avg_metrics:
    avg_metrics[metric] /= len(results)

# 출력
print("\n🔍 Evaluation Results:")
for metric, value in sorted(avg_metrics.items()):
    print(f"{metric:12}: {value:.4f}")
