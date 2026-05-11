import re
import string
import time
from collections import Counter


def normalize_text(text):
    """
    Metni küçük harfe çevirir, noktalama işaretlerini kaldırır,
    ekstra boşlukları temizler.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# ANSWER METRICS
# =========================================================
def exact_match_score(prediction, ground_truth):
    """
    Tahmin ile gerçek cevap birebir aynı mı?
    """
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def f1_score(prediction, ground_truth):
    """
    Token-level F1 score hesaplar.
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


# =========================================================
# RETRIEVAL METRICS
# =========================================================
def context_precision(retrieved_chunks, relevant_chunks):
    """
    Çekilen chunk'ların ne kadarı gerçekten ilgili?
    """
    if not retrieved_chunks:
        return 0.0
    if not relevant_chunks:
        return 0.0

    retrieved_norm = [normalize_text(c) for c in retrieved_chunks]
    relevant_norm = [normalize_text(c) for c in relevant_chunks]

    relevant_retrieved = sum(1 for chunk in retrieved_norm if chunk in relevant_norm)
    return relevant_retrieved / len(retrieved_norm)


def context_recall(retrieved_chunks, relevant_chunks):
    """
    Gerçek relevant chunk'ların ne kadarı çekildi?
    """
    if not relevant_chunks:
        return 0.0
    if not retrieved_chunks:
        return 0.0

    retrieved_norm = [normalize_text(c) for c in retrieved_chunks]
    relevant_norm = [normalize_text(c) for c in relevant_chunks]

    found_relevant = sum(1 for chunk in relevant_norm if chunk in retrieved_norm)
    return found_relevant / len(relevant_norm)


def hit_at_k(retrieved_chunks, relevant_chunks, k=3):
    """
    İlk k sonuç içinde en az bir doğru chunk var mı?
    """
    if not retrieved_chunks or not relevant_chunks:
        return 0.0

    top_k = [normalize_text(c) for c in retrieved_chunks[:k]]
    relevant_norm = [normalize_text(c) for c in relevant_chunks]

    for chunk in top_k:
        if chunk in relevant_norm:
            return 1.0
    return 0.0


def reciprocal_rank(retrieved_chunks, relevant_chunks):
    """
    İlk doğru sonucun sırasına göre skor verir.
    İlk doğru sonuç 1. sıradaysa 1.0
    2. sıradaysa 0.5
    3. sıradaysa 0.333...
    """
    if not retrieved_chunks or not relevant_chunks:
        return 0.0

    relevant_norm = [normalize_text(c) for c in relevant_chunks]

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        if normalize_text(chunk) in relevant_norm:
            return 1.0 / idx
    return 0.0


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def get_category(item):
    """
    Dataset item içinden kategori bilgisini döndürür.
    """
    return item.get("category", "Uncategorized")


def measure_latency(start_time):
    """
    Başlangıç zamanına göre geçen süreyi saniye cinsinden döndürür.
    """
    return round(time.time() - start_time, 4)