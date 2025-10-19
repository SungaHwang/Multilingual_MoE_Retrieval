# 세 메트릭(P@1, MRR, nDCG@2)에 대한 평균 성능을 전문가 수에 따라 시각화
# 데이터 수동 추출 (from provided image)
import matplotlib.pyplot as plt
expert_counts = [2, 3, 4, 5, 6]

# 각 메트릭별 모델 성능 리스트 (w/o lang, w/ lang)
wo_p = [0.3619, 0.3789, 0.3566, 0.3681, 0.3510]
wo_mrr = [0.4985, 0.5143, 0.5047, 0.5065, 0.4892]
wo_ndcg = [0.4341, 0.4455, 0.4369, 0.4393, 0.4178]

w_p = [0.3615, 0.3830, 0.3623, 0.3700, 0.3621]
w_mrr = [0.5009, 0.5185, 0.5169, 0.5092, 0.5040]
w_ndcg = [0.4301, 0.4514, 0.4518, 0.4335, 0.4335]

# 시각화
plt.figure(figsize=(8, 6))

# P@1
plt.plot(expert_counts, wo_p, label='P@1 (w/o lang)', marker='o', color='orange')
plt.plot(expert_counts, w_p, label='P@1 (w/ lang)', marker='o', color='darkorange', linestyle='--')

# MRR
plt.plot(expert_counts, wo_mrr, label='MRR@10 (w/o lang)', marker='s', color='blue')
plt.plot(expert_counts, w_mrr, label='MRR@10 (w/ lang)', marker='s', color='darkblue', linestyle='--')

# nDCG@2
plt.plot(expert_counts, wo_ndcg, label='nDCG@2 (w/o lang)', marker='^', color='green')
plt.plot(expert_counts, w_ndcg, label='nDCG@2 (w/ lang)', marker='^', color='darkgreen', linestyle='--')

plt.title("Average Performance Variation by Number of Experts in Multilingual Retrieval")
plt.xlabel("Number of Experts")
plt.ylabel("Score")
plt.xticks(expert_counts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
