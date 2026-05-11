import json
import os
import sys
import csv
import time
from datetime import datetime

# =========================================================
# 1) KLASÖR YOLLARI
# =========================================================
current_file_path = os.path.abspath(__file__)
benchmark_dir = os.path.dirname(current_file_path)
base_dir = os.path.dirname(benchmark_dir)

if base_dir not in sys.path:
    sys.path.append(base_dir)

# =========================================================
# 2) MODÜLLERİ YÜKLE
# =========================================================
from benchmark.metrics import (
    context_recall,
    context_precision,
    reciprocal_rank,
    f1_score,
    hit_at_k,
    exact_match_score,
    get_category,
    measure_latency,
)

from rag_system.retriever import retriever
from rag_system.generator import generator

# =========================================================
# 3) DOSYA YOLLARI
# =========================================================
DATASET_PATH = os.path.join(benchmark_dir, "dataset.json")
RESULTS_DIR = os.path.join(base_dir, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "leaderboard.csv")


# =========================================================
# 4) YARDIMCI FONKSİYONLAR
# =========================================================
def safe_json_dumps(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "[]"


def extract_retrieved_texts(retrieved_docs):
    texts = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            text = doc.get("text", "")
            if text:
                texts.append(text)
        else:
            texts.append(str(doc))
    return texts


def extract_retrieved_chunk_ids(retrieved_docs):
    chunk_ids = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            chunk_id = doc.get("chunk_id", "")
            if chunk_id:
                chunk_ids.append(chunk_id)
    return chunk_ids


# =========================================================
# 5) ANA BENCHMARK FONKSİYONU
# =========================================================
def run_benchmark():
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    top_k = int(os.getenv("TOP_K", 5))

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tarih": datetime.now().strftime("%Y-%m-%d"),
        "model": os.getenv("CURRENT_MODEL", "Llama-3.3-70b"),
        "retriever": os.getenv("CURRENT_RETRIEVER", "Keyword-Based"),
        "prompt_v": os.getenv("PROMPT_VERSION", "v1.0.2"),
        "chunk_v": os.getenv("CHUNK_VERSION", "500-char-overlap"),
        "top_k": top_k,
    }

    if not os.path.exists(DATASET_PATH):
        print(f" HATA: Dataset bulunamadı: {DATASET_PATH}")
        return

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            print(" HATA: dataset.json liste formatında olmalı.")
            return

    except Exception as e:
        print(f" Dataset okuma hatası: {e}")
        return

    results = []

    print(
        f"[INFO] Benchmark başladı | run_id={run_id} | "
        f"model={metadata['model']} | retriever={metadata['retriever']} | top_k={top_k}"
    )

    for idx, item in enumerate(dataset, start=1):
        question = item.get("question", "")
        ground_truth = item.get("gold_answer", "")
        source = item.get("source_passage", "")
        category = get_category(item)

        doc_id = item.get("doc_id", "")
        question_id = item.get("id", f"q_{idx:03d}")
        gold_chunk_ids = item.get("gold_chunk_ids", [])

        relevant_chunks = [source] if source else []

        try:
            # 1) Retrieval
            retrieved_docs = retriever.search(question, k=top_k)
            retrieved_texts = extract_retrieved_texts(retrieved_docs)
            retrieved_chunk_ids = extract_retrieved_chunk_ids(retrieved_docs)

            # 2) Generation + latency
            start_time = time.time()
            answer = generator.generate(retrieved_docs, question)
            latency = measure_latency(start_time)

            # 3) Retrieval metrics
            rec = context_recall(retrieved_texts, relevant_chunks)
            pre = context_precision(retrieved_texts, relevant_chunks)
            hit_val = hit_at_k(retrieved_texts, relevant_chunks, k=min(3, top_k))
            mrr_val = reciprocal_rank(retrieved_texts, relevant_chunks)

            # 4) Answer metrics
            f1_val = f1_score(answer, ground_truth)
            em_val = exact_match_score(answer, ground_truth)

            # 5) Save result
            results.append({
                **metadata,
                "question_id": question_id,
                "doc_id": doc_id,
                "question": question,
                "generated_answer": answer,
                "ground_truth": ground_truth,
                "category": category,
                "latency": latency,

                "gold_chunk_ids": safe_json_dumps(gold_chunk_ids),
                "retrieved_chunk_ids": safe_json_dumps(retrieved_chunk_ids),
                "retrieved_context": safe_json_dumps(retrieved_texts),

                "recall": round(rec, 4),
                "precision": round(pre, 4),
                "mrr": round(mrr_val, 4),
                "hit_at_k": round(hit_val, 4),

                "f1": round(f1_val, 4),
                "exact_match": round(em_val, 4),

                # Geçici yaklaşık skorlar
                "hallucination": round(1 - pre, 4),
                "faithfulness": round(pre, 4),

                "status": "success",
                "error_message": ""
            })

            print(f"[INFO] [{idx}/{len(dataset)}] Tamamlandı: {question[:60]}")

        except Exception as e:
            error_msg = str(e)

            results.append({
                **metadata,
                "question_id": question_id,
                "doc_id": doc_id,
                "question": question,
                "generated_answer": "",
                "ground_truth": ground_truth,
                "category": category,
                "latency": None,

                "gold_chunk_ids": safe_json_dumps(gold_chunk_ids),
                "retrieved_chunk_ids": safe_json_dumps([]),
                "retrieved_context": safe_json_dumps([]),

                "recall": 0.0,
                "precision": 0.0,
                "mrr": 0.0,
                "hit_at_k": 0.0,

                "f1": 0.0,
                "exact_match": 0.0,

                "hallucination": None,
                "faithfulness": None,

                "status": "error",
                "error_message": error_msg
            })

            print(f" [{idx}/{len(dataset)}] Hata: {question[:60]} -> {error_msg}")

    # =========================================================
    # 6) SONUÇLARI CSV'YE YAZ
    # =========================================================
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    file_exists = os.path.isfile(RESULTS_PATH)

    if results:
        with open(RESULTS_PATH, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=results[0].keys(),
                quoting=csv.QUOTE_MINIMAL
            )

            if not file_exists:
                writer.writeheader()

            writer.writerows(results)

        print(f"[INFO] Başarıyla kaydedildi: {len(results)} satır -> {RESULTS_PATH}")
    else:
        print("[HATA] Kaydedilecek sonuç bulunamadı.")


if __name__ == "__main__":
    run_benchmark()