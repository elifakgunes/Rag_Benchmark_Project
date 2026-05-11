# Rag_Benchmark_Project
RAG Benchmark & Evaluation System

Bu proje, Retrieval-Augmented Generation (RAG) sistemlerinin performansını ölçmek için geliştirilmiş bir benchmark ve değerlendirme platformudur. Sistem; dokümanlardan bilgi çekme (retrieval), LLM ile cevap üretme (generation) ve üretilen cevapların doğruluğunu farklı metriklerle değerlendirme süreçlerini içerir.

Projede ayrıca Streamlit tabanlı bir dashboard bulunmaktadır. Bu dashboard üzerinden:

Dataset yönetimi
Benchmark çalıştırma
Model karşılaştırma
Geçmiş sonuçları analiz etme
işlemleri yapılabilir.

Projenin Amacı

RAG sistemleri genellikle şu sorunlarla karşılaşır:

yanlış doküman getirme (retrieval error)
eksik bağlam
halüsinasyon (hallucination)
model cevaplarının doğruluğunu ölçememe
Bu proje şu problemleri çözmeyi amaçlar:

RAG sistemlerini standart bir benchmark dataset ile test etmek
farklı retriever ve model kombinasyonlarını karşılaştırmak
halüsinasyon oranını ölçmek
performansı dashboard üzerinden analiz etmek
Proje Yapısı

Rag_project
│
├── rag_system
│   ├── retriever.py
│   ├── generator.py
│
├── benchmark
│   ├── evaluate.py
│   ├── metrics.py
│   ├── dataset.json
│
├── corpus
│   ├── corpus_chunks.json
│
├── app.py
│
├── requirements.txt
Sistem Mimarisi

Soru
   │
   ▼
Retriever
   │
   ▼
Relevant Chunks
   │
   ▼
Generator (LLM)
   │
   ▼
Cevap
   │
   ▼
Evaluation Metrics
Adımlar:

Dataset içinden bir soru alınır
Retriever ilgili doküman parçalarını bulur
Generator modeli bağlam ile birlikte cevap üretir
Cevap benchmark metrikleri ile değerlendirilir
Sonuçlar dashboard'a kaydedilir
Kurulum

Projeyi klonlayın:

git clone https://github.com/kullaniciadi/Rag_project.git
cd Rag_project
Python Ortamı Oluşturma

Windows:

python -m venv .venv
.venv\Scripts\activate
Mac / Linux:

python3 -m venv .venv
source .venv/bin/activate
Gerekli Paketler

Projede kullanılan temel kütüphaneler:

streamlit
pandas
numpy
sentence-transformers
scikit-learn
google-generativeai
chromadb
python-dotenv
plotly
Kurmak için:

pip install -r requirements.txt
veya

pip install streamlit pandas numpy sentence-transformers scikit-learn google-generativeai chromadb python-dotenv plotly
API Anahtarı Ayarlama

Proje Gemini gibi bir LLM kullanıyorsa .env dosyası oluşturulmalıdır.

GEMINI_API_KEY=your_api_key_here
Corpus Yapısı

Retriever sisteminin çalışabilmesi için dokümanlar chunklara bölünür.

corpus/corpus_chunks.json
Örnek veri:

[
  {
    "doc_id": "mam_pdf",
    "chunk_id": "mam_pdf_chunk_001",
    "text": "TÜBİTAK Marmara Araştırma Merkezi 1972 yılında kurulmuştur.",
    "page": 3
  }
]
Dataset Yapısı

Benchmark soruları şu dosyada bulunur:

benchmark/dataset.json
Örnek:

{
  "id": "q_001",
  "question": "TÜBİTAK MAM hangi yıl kurulmuştur?",
  "gold_answer": "1972",
  "source_passage": "TÜBİTAK Marmara Araştırma Merkezi 1972 yılında kurulmuştur.",
  "category": "Faktüel",
  "difficulty": "easy",
  "answerable": true
}
Benchmark Çalıştırma

Benchmark başlatmak için:

python -m benchmark.evaluate
Bu işlem:

dataset içindeki soruları işler
retriever ile ilgili dokümanları getirir
generator model ile cevap üretir
metrikleri hesaplar
sonuçları kaydeder
Dashboard Çalıştırma

Streamlit dashboard'u çalıştırmak için:

streamlit run app.py
Dashboard üzerinden:

Dataset yönetimi
Benchmark sonuçlarını görüntüleme
Model karşılaştırma
geçmiş benchmark runları
analiz edilebilir.

Ölçülen Metrikler

Retrieval Metrics

Recall
Precision
Context Recall
Generation Metrics

Exact Match
F1 Score
Semantic Similarity
Hallucination Metrics

Modelin dokümanda olmayan bilgi üretme oranı ölçülür.

Hallucination Testi

Dataset içinde answerable = false olan sorular halüsinasyon testleri için kullanılır.

Örnek:

TÜBİTAK MAM’ın yıllık bütçesi ne kadardır?
Model cevap uydurursa:

Hallucination
olarak işaretlenir.

Kullanılan Teknolojiler

Python
Streamlit
Sentence Transformers
ChromaDB
Gemini / LLM
Pandas
Plotly
