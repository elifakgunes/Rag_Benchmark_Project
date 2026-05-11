import json
import pandas as pd
from metrics import context_recall, context_precision, exact_match, f1_score, semantic_similarity
from rag_system.retriever import retriever
from rag_system.generator import generator

# 1. Dataset yükle
with open("dataset.json") as f:
    dataset = json.load(f)

results = []

# 2. Sorular üzerinde dön
for item in dataset:
    question = item["question"]
    ground_truth = item["answer"]
    relevant_chunks = item.get("relevant_chunks", [])

    # 3. Retriever çalıştır
    retrieved_docs = retriever.search(question, k=5)

    # 4. Generator çalıştır
    answer = generator.generate(retrieved_docs, question)

    # 5. Metric hesapla
    recall = context_recall(retrieved_docs, relevant_chunks)
    precision = context_precision(retrieved_docs, relevant_chunks)
    em = exact_match(answer, ground_truth)
    f1 = f1_score(answer, ground_truth)
    sim = semantic_similarity(answer, ground_truth)

    results.append({
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "recall": recall,
        "precision": precision,
        "exact_match": em,
        "f1": f1,
        "semantic_similarity": sim
    })

# 6. CSV olarak kaydet
df = pd.DataFrame(results)
df.to_csv("results/leaderboard.csv", index=False)
print("Benchmark tamamlandı! results/leaderboard.csv oluşturuldu.")