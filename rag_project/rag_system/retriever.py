import json
import os
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, retriever_type: str = "Keyword-Based"):
        self.retriever_type = retriever_type
        self.documents = self.load_corpus()
        self.embedding_model = None
        self.document_embeddings = None

        # Semantic ve Hybrid için embedding modeli yükle
        if self.retriever_type in ["Semantic-Vector", "Hybrid-Search"] and self.documents:
            self._initialize_embeddings()

    def _get_base_dir(self) -> str:
        """
        Proje kök klasörünü bulmaya çalışır.
        Eğer retriever.py proje kökünde ise current_dir kullanır.
        Eğer bir alt klasörde ise bir üst klasörü dener.
        """
        current_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_path)

        # Olası corpus yolları
        candidate_paths = [
            os.path.join(current_dir, "data", "corpus_chunks.json"),
            os.path.join(os.path.dirname(current_dir), "data", "corpus_chunks.json"),
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                return os.path.dirname(os.path.dirname(path)) if path.endswith(os.path.join("data", "corpus_chunks.json")) else current_dir

        # Varsayılan: dosyanın bulunduğu klasör
        return current_dir

    def load_corpus(self) -> List[Dict[str, Any]]:
        """
        data/corpus_chunks.json dosyasını okur.
        Beklenen format:
        [
          {
            "doc_id": "mam_pdf",
            "chunk_id": "mam_pdf_chunk_0012",
            "text": "....",
            "page": 3,
            "source_file": "mam.pdf"
          }
        ]
        """
        current_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_path)

        possible_paths = [
            os.path.join(current_dir, "data", "corpus_chunks.json"),
            os.path.join(os.path.dirname(current_dir), "data", "corpus_chunks.json"),
        ]

        corpus_path = None
        for path in possible_paths:
            if os.path.exists(path):
                corpus_path = path
                break

        if corpus_path is None:
            print("[Hata] Corpus bulunamadı. Beklenen yol: data/corpus_chunks.json")
            return []

        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)

            if not isinstance(corpus, list):
                print("[Hata] corpus_chunks.json liste formatında değil.")
                return []

            docs = []
            for item in corpus:
                if not isinstance(item, dict):
                    continue

                text = item.get("text", "")
                if not text or not isinstance(text, str):
                    continue

                docs.append({
                    "doc_id": item.get("doc_id", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "text": text,
                    "page": item.get("page", None),
                    "source_file": item.get("source_file", "")
                })

            print(f" Corpus yüklendi: {len(docs)} chunk")
            return docs

        except Exception as e:
            print(f" Corpus okuma hatası: {e}")
            return []

    def _initialize_embeddings(self) -> None:
        """
        Semantic / Hybrid için embedding modelini ve doküman embedding'lerini hazırlar.
        """
        try:
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            texts = [doc["text"] for doc in self.documents]
            self.document_embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            print("Embedding modeli hazır.")
        except Exception as e:
            print(f"[Hata] Embedding modeli yüklenemedi: {e}")
            self.embedding_model = None
            self.document_embeddings = None

    def preprocess_query(self, query: str) -> List[str]:
        """
        Basit query temizleme ve stopword filtreleme.
        """
        query = query.lower().replace("?", "").replace(".", "").replace(",", "").replace("!", "").strip()

        stop_words = {
            "ve", "veya", "da", "de", "için", "olan", "bir", "ile",
            "bu", "şu", "o", "mi", "mı", "mu", "mü",
            "nedir", "hangi", "kaç", "kim", "nasıl", "ne",
            "tübitak", "mam", "marmara", "araştırma", "merkezi"
        }

        words = [w for w in query.split() if w not in stop_words and len(w) > 2]
        return words if words else query.split()

    def keyword_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Kelime eşleşmesine dayalı basit retrieval.
        """
        search_words = self.preprocess_query(query)
        scored_docs = []

        for doc in self.documents:
            doc_text = doc["text"].lower()
            score = sum(1 for word in search_words if word in doc_text)

            if score > 0:
                scored_docs.append({
                    **doc,
                    "keyword_score": float(score)
                })

        scored_docs.sort(key=lambda x: x["keyword_score"], reverse=True)
        return scored_docs[:k]

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Embedding + cosine similarity tabanlı retrieval.
        """
        if self.embedding_model is None or self.document_embeddings is None:
            print("[Hata] Semantic retriever hazır değil, keyword search'e düşülüyor.")
            return self.keyword_search(query, k)

        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]

            scored_docs = []
            for doc, sim in zip(self.documents, similarities):
                scored_docs.append({
                    **doc,
                    "semantic_score": float(sim)
                })

            scored_docs.sort(key=lambda x: x["semantic_score"], reverse=True)
            return scored_docs[:k]

        except Exception as e:
            print(f"[Hata] Semantic search hatası: {e}")
            return self.keyword_search(query, k)

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Keyword + Semantic birleşik retrieval.
        Basit birleştirme:
        hybrid_score = 0.5 * normalized_keyword + 0.5 * semantic_score
        """
        if self.embedding_model is None or self.document_embeddings is None:
            print("[Hata] Hybrid retriever hazır değil, keyword search'e düşülüyor.")
            return self.keyword_search(query, k)

        try:
            search_words = self.preprocess_query(query)
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]

            scored_docs = []
            max_keyword_score = max(len(search_words), 1)

            for doc, sim in zip(self.documents, similarities):
                doc_text = doc["text"].lower()
                keyword_score = sum(1 for word in search_words if word in doc_text)
                normalized_keyword = keyword_score / max_keyword_score

                hybrid_score = 0.5 * normalized_keyword + 0.5 * float(sim)

                scored_docs.append({
                    **doc,
                    "keyword_score": float(keyword_score),
                    "semantic_score": float(sim),
                    "hybrid_score": float(hybrid_score)
                })

            scored_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
            return scored_docs[:k]

        except Exception as e:
            print(f"[Hata] Hybrid search hatası: {e}")
            return self.keyword_search(query, k)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retriever tipine göre uygun yöntemi çağırır.
        """
        if not self.documents:
            return []

        if self.retriever_type == "Keyword-Based":
            return self.keyword_search(query, k)

        if self.retriever_type == "Semantic-Vector":
            return self.semantic_search(query, k)

        if self.retriever_type == "Hybrid-Search":
            return self.hybrid_search(query, k)

        # Bilinmeyen bir değer gelirse fallback
        return self.keyword_search(query, k)


selected_retriever = os.getenv("CURRENT_RETRIEVER", "Keyword-Based")
retriever = Retriever(retriever_type=selected_retriever)