import os
import json
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import sys
sys.stdout.reconfigure(encoding='utf-8')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RAG Benchmark Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS
# =========================================================
CURRENT_FILE = os.path.abspath(__file__)
DASHBOARD_DIR = os.path.dirname(CURRENT_FILE)
BASE_DIR = os.path.dirname(DASHBOARD_DIR)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "leaderboard.csv")
DATASET_PATH = os.path.join(BASE_DIR, "benchmark", "dataset.json")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus_chunks.json")

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
        .main > div {
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px;
            border-radius: 16px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
        }
        .small-muted {
            color: #9ca3af;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def safe_read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    numeric_cols = [
        "f1", "exact_match", "recall", "precision", "mrr", "hit_at_k",
        "hallucination", "faithfulness", "latency", "top_k"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "tarih" in df.columns:
        df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")

    return df


@st.cache_data
def load_dataset():
    data = safe_read_json(DATASET_PATH, [])
    return data if isinstance(data, list) else []


@st.cache_data
def load_corpus():
    data = safe_read_json(CORPUS_PATH, [])
    return data if isinstance(data, list) else []


def parse_json_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [str(value)] if str(value).strip() else []


def summarize_runs(df):
    if df.empty or "run_id" not in df.columns:
        return pd.DataFrame()

    metric_cols = [c for c in [
        "f1", "exact_match", "recall", "precision",
        "mrr", "hit_at_k", "hallucination", "faithfulness", "latency"
    ] if c in df.columns]

    summary = df.groupby("run_id", as_index=False)[metric_cols].mean() if metric_cols else df[["run_id"]].drop_duplicates()

    meta_cols = [c for c in [
        "run_id", "timestamp", "tarih", "model", "retriever",
        "prompt_v", "chunk_v", "top_k"
    ] if c in df.columns]

    if meta_cols:
        meta_df = df[meta_cols].drop_duplicates(subset=["run_id"], keep="last")
        summary = summary.merge(meta_df, on="run_id", how="left")

    question_counts = df.groupby("run_id", as_index=False).size().rename(columns={"size": "question_count"})
    summary = summary.merge(question_counts, on="run_id", how="left")

    if "status" in df.columns:
        error_df = (
            df.assign(_is_error=df["status"].astype(str).eq("error"))
              .groupby("run_id", as_index=False)["_is_error"]
              .sum()
              .rename(columns={"_is_error": "error_count"})
        )
        summary = summary.merge(error_df, on="run_id", how="left")
        summary["success_count"] = summary["question_count"] - summary["error_count"]
        summary["success_rate"] = (summary["success_count"] / summary["question_count"]).fillna(0)

    if "timestamp" in summary.columns:
        summary = summary.sort_values("timestamp", ascending=False)
    elif "tarih" in summary.columns:
        summary = summary.sort_values("tarih", ascending=False)

    return summary


def dataset_health(dataset):
    total = len(dataset)
    categories = {}
    doc_counts = {}
    difficulty_counts = {}
    answerable_yes = 0
    approved_yes = 0

    for item in dataset:
        cat = item.get("category", "Uncategorized")
        categories[cat] = categories.get(cat, 0) + 1

        doc_id = item.get("doc_id", "unknown")
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        diff = item.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        if item.get("answerable", True):
            answerable_yes += 1

        if item.get("approved", False):
            approved_yes += 1

    return {
        "total": total,
        "categories": categories,
        "doc_counts": doc_counts,
        "difficulty_counts": difficulty_counts,
        "answerable_yes": answerable_yes,
        "answerable_no": total - answerable_yes,
        "approved_yes": approved_yes,
        "approved_no": total - approved_yes,
    }


def metric_row(df):
    c1, c2, c3, c4 = st.columns(4)

    avg_f1 = round(df["f1"].mean(), 4) if "f1" in df.columns and not df["f1"].dropna().empty else 0.0
    avg_recall = round(df["recall"].mean(), 4) if "recall" in df.columns and not df["recall"].dropna().empty else 0.0
    avg_hall = round(df["hallucination"].mean(), 4) if "hallucination" in df.columns and not df["hallucination"].dropna().empty else 0.0
    avg_latency = round(df["latency"].mean(), 4) if "latency" in df.columns and not df["latency"].dropna().empty else 0.0

    c1.metric("Avg F1", avg_f1)
    c2.metric("Avg Recall", avg_recall)
    c3.metric("Avg Hallucination", avg_hall)
    c4.metric("Avg Latency", avg_latency)


def export_bytes(df, kind="csv"):
    if kind == "csv":
        return df.to_csv(index=False).encode("utf-8-sig")
    return df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")


def run_benchmark_from_dashboard(model_name, retriever_name, prompt_v, chunk_v, top_k):
    env = os.environ.copy()
    env["CURRENT_MODEL"] = model_name
    env["CURRENT_RETRIEVER"] = retriever_name
    env["PROMPT_VERSION"] = prompt_v
    env["CHUNK_VERSION"] = chunk_v
    env["TOP_K"] = str(top_k)

    command = [sys.executable, "-m", "benchmark.evaluate"]
    return subprocess.run(
        command,
        cwd=BASE_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data):
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        return []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_doc_ids_from_corpus(corpus_json):
    doc_ids = []
    for item in corpus_json:
        doc_id = item.get("doc_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def get_chunks_by_doc_id(corpus_json, doc_id):
    return [x for x in corpus_json if x.get("doc_id") == doc_id]


def generate_question_id(dataset):
    ids = [int(x["id"].split("_")[1]) for x in dataset if "id" in x]
    next_id = max(ids, default=0) + 1
    return f"q_{next_id:03d}"
def add_manual_question(
    dataset,
    question,
    gold_answer,
    source_passage,
    category,
    difficulty,
    answerable,
    approved,
    doc_id,
    gold_chunk_ids
):
    new_id_num = len(dataset) + 1

    # q_001, q_002 gibi id üret
    existing_ids = []
    for item in dataset:
        item_id = str(item.get("id", ""))
        if item_id.startswith("q_"):
            try:
                existing_ids.append(int(item_id.split("_")[1]))
            except:
                pass

    if existing_ids:
        new_id_num = max(existing_ids) + 1

    new_item = {
        "id": f"q_{new_id_num:03d}",
        "question": question,
        "gold_answer": gold_answer,
        "source_passage": source_passage,
        "category": category,
        "difficulty": difficulty,
        "answerable": answerable,
        "approved": approved,
        "doc_id": doc_id,
        "gold_chunk_ids": gold_chunk_ids,
        "created_by": "manual",
        "created_at": datetime.now().isoformat()
    }

    dataset.append(new_item)
    save_dataset(dataset)
    return new_item
def simple_candidate_generation_from_chunks(chunks, n_questions=5):
    candidates = []

    for i, ch in enumerate(chunks[:n_questions], start=1):
        chunk_text = str(ch.get("text", "")).strip()
        chunk_id = ch.get("chunk_id", "")

        if not chunk_text:
            continue

        short_text = chunk_text[:400]

        candidate = {
            "question": f"Bu parçaya göre temel bilgi nedir? ({i})",
            "gold_answer": short_text[:200],
            "source_passage": short_text,
            "gold_chunk_ids": [chunk_id] if chunk_id else []
        }

        candidates.append(candidate)

    return candidates
def add_candidate_questions(dataset, candidates, doc_id, category, difficulty):
    created = []

    existing_ids = []
    for item in dataset:
        item_id = str(item.get("id", ""))
        if item_id.startswith("q_"):
            try:
                existing_ids.append(int(item_id.split("_")[1]))
            except:
                pass

    next_id = max(existing_ids, default=0) + 1

    for cand in candidates:
        new_item = {
            "id": f"q_{next_id:03d}",
            "question": cand.get("question", ""),
            "gold_answer": cand.get("gold_answer", ""),
            "source_passage": cand.get("source_passage", ""),
            "category": category,
            "difficulty": difficulty,
            "answerable": True,
            "approved": False,
            "doc_id": doc_id,
            "gold_chunk_ids": cand.get("gold_chunk_ids", []),
            "created_by": "llm_candidate",
            "created_at": datetime.now().isoformat()
        }

        dataset.append(new_item)
        created.append(new_item)
        next_id += 1

    save_dataset(dataset)
    return created


# =========================================================
# LOAD DATA
# =========================================================
results_df = load_results()
dataset_json = load_dataset()
corpus_json = load_corpus()
run_summary_df = summarize_runs(results_df) if not results_df.empty else pd.DataFrame()
health = dataset_health(dataset_json)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📊 RAG Benchmark Platform")

page = st.sidebar.radio(
    "Menü",
    [
        "🏠 Overview",
        "🚀 Benchmark Runner",
        "🗂️ Dataset Manager",
        "🏆 Leaderboard",
        "🆚 Run Comparison",
        "❌ Error Analysis",
        "🧩 Question Trace",
        "📚 Dataset Analytics",
        "💾 Export",
    ]
)

# Global filters
st.sidebar.markdown("---")
st.sidebar.markdown("### Global Filters")

filtered_df = results_df.copy()

if not filtered_df.empty:
    if "model" in filtered_df.columns:
        model_options = ["All"] + sorted(filtered_df["model"].dropna().astype(str).unique().tolist())
        selected_model = st.sidebar.selectbox("Model", model_options, index=0)
        if selected_model != "All":
            filtered_df = filtered_df[filtered_df["model"].astype(str) == selected_model]

    if "retriever" in filtered_df.columns:
        retriever_options = ["All"] + sorted(filtered_df["retriever"].dropna().astype(str).unique().tolist())
        selected_retriever = st.sidebar.selectbox("Retriever", retriever_options, index=0)
        if selected_retriever != "All":
            filtered_df = filtered_df[filtered_df["retriever"].astype(str) == selected_retriever]

    if "category" in filtered_df.columns:
        category_options = ["All"] + sorted(filtered_df["category"].dropna().astype(str).unique().tolist())
        selected_category = st.sidebar.selectbox("Category", category_options, index=0)
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["category"].astype(str) == selected_category]

# =========================================================
# HEADER
# =========================================================
st.title("RAG Benchmark Platform")
st.caption("Tekrarlanabilir, karşılaştırılabilir ve versiyonlanabilir RAG benchmark dashboard")

# =========================================================
# 1. OVERVIEW
# =========================================================
if page == "🏠 Overview":
    if filtered_df.empty:
        st.warning("Henüz sonuç verisi bulunamadı. Önce benchmark çalıştır.")
    else:
        metric_row(filtered_df)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Toplam Run", filtered_df["run_id"].nunique() if "run_id" in filtered_df.columns else 0)
        c2.metric("Toplam Soru", len(filtered_df))
        c3.metric("Ort. F1", round(filtered_df["f1"].mean(), 4) if "f1" in filtered_df.columns else 0)
        c4.metric("Ort. Recall", round(filtered_df["recall"].mean(), 4) if "recall" in filtered_df.columns else 0)
        c5.metric("Ort. Hallucination", round(filtered_df["hallucination"].mean(), 4) if "hallucination" in filtered_df.columns else 0)
        c6.metric("Ort. Latency", round(filtered_df["latency"].mean(), 4) if "latency" in filtered_df.columns else 0)

        st.markdown("---")

        left, right = st.columns([1.4, 1])

        with left:
            st.markdown("### Run Bazlı Trend Grafiği")
            if not run_summary_df.empty:
                trend_cols = [c for c in ["f1", "recall", "precision", "hallucination", "latency"] if c in run_summary_df.columns]
                y_metric = st.selectbox("Trend metriği", trend_cols, index=0)
                x_col = "timestamp" if "timestamp" in run_summary_df.columns else "run_id"
                fig = px.line(
                    run_summary_df.sort_values(x_col),
                    x=x_col,
                    y=y_metric,
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run summary verisi yok.")

        with right:
            st.markdown("### Last Run Summary")
            if not run_summary_df.empty:
                last_run = run_summary_df.iloc[0]
                st.json({
                    "run_id": str(last_run.get("run_id", "")),
                    "timestamp": str(last_run.get("timestamp", last_run.get("tarih", ""))),
                    "model": str(last_run.get("model", "")),
                    "retriever": str(last_run.get("retriever", "")),
                    "total_questions": int(last_run.get("question_count", 0)) if pd.notna(last_run.get("question_count", None)) else 0,
                    "success_rate": round(float(last_run.get("success_rate", 0)), 4) if pd.notna(last_run.get("success_rate", None)) else 0,
                    "avg_f1": round(float(last_run.get("f1", 0)), 4) if pd.notna(last_run.get("f1", None)) else 0,
                    "avg_recall": round(float(last_run.get("recall", 0)), 4) if pd.notna(last_run.get("recall", None)) else 0,
                    "avg_latency": round(float(last_run.get("latency", 0)), 4) if pd.notna(last_run.get("latency", None)) else 0,
                })
            else:
                st.info("Henüz run yok.")

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Model Bazlı Ortalama Performans")
            if "model" in filtered_df.columns:
                model_summary = (
                    filtered_df.groupby("model", as_index=False)[
                        [c for c in ["f1", "recall", "precision", "hallucination"] if c in filtered_df.columns]
                    ].mean()
                )
                fig = px.bar(
                    model_summary,
                    x="model",
                    y=[c for c in ["f1", "recall", "precision", "hallucination"] if c in model_summary.columns],
                    barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model kolonu bulunamadı.")

        with c2:
            st.markdown("### Retriever Bazlı Ortalama Performans")
            if "retriever" in filtered_df.columns:
                retr_summary = (
                    filtered_df.groupby("retriever", as_index=False)[
                        [c for c in ["f1", "recall", "precision", "hallucination"] if c in filtered_df.columns]
                    ].mean()
                )
                fig = px.bar(
                    retr_summary,
                    x="retriever",
                    y=[c for c in ["f1", "recall", "precision", "hallucination"] if c in retr_summary.columns],
                    barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Retriever kolonu bulunamadı.")

        st.markdown("---")

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("### Latency Histogram")
            if "latency" in filtered_df.columns:
                fig = px.histogram(filtered_df, x="latency", nbins=25)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Latency kolonu yok.")

        with c4:
            st.markdown("### Metric Dağılımı")
            metric_options = [c for c in ["f1", "recall", "precision", "hallucination", "latency"] if c in filtered_df.columns]
            if metric_options:
                selected_metric = st.selectbox("Box plot metriği", metric_options, index=0)
                group_by = "model" if "model" in filtered_df.columns else "retriever"
                fig = px.box(filtered_df, x=group_by, y=selected_metric)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Metric kolonu yok.")

# =========================================================
# 2. BENCHMARK RUNNER
# =========================================================
elif page == "🚀 Benchmark Runner":
    st.subheader("Benchmark Runner")
    metric_row(filtered_df if not filtered_df.empty else pd.DataFrame())

    left, right = st.columns([1.1, 1])

    with left:
        model_name = st.selectbox(
            "Model Seç",
            ["Llama-3.3-70b", "Llama-3.1-8b", "GPT-4o", "GPT-4o-mini", "Gemini Pro", "Gemini Flash"],
            index=0
        )

        retriever_name = st.selectbox(
            "Retriever Seç",
            ["Keyword-Based", "Semantic-Vector", "Hybrid-Search"],
            index=0
        )

        prompt_v = st.selectbox(
            "Prompt Version",
            ["v1.0.0", "v1.0.1", "v1.0.2", "v2.0.0", "v3.0.0"],
            index=2
        )

        chunk_v = st.selectbox(
            "Chunk Version",
            ["300-char", "500-char-overlap", "800-char", "semantic-chunks"],
            index=1
        )

        top_k = st.slider("Top-K", 1, 10, 5)

    with right:
        st.markdown("### Dataset Bilgisi")
        st.metric("Question Count", health["total"])
        st.metric("Corpus Chunk Count", len(corpus_json))
        st.metric("Answerable", health["answerable_yes"])
        st.metric("Unanswerable", health["answerable_no"])

        st.markdown("### Çalıştırılacak Konfigürasyon")
        st.json({
            "model": model_name,
            "retriever": retriever_name,
            "prompt_v": prompt_v,
            "chunk_v": chunk_v,
            "top_k": top_k,
        })

    st.markdown("---")

    if st.button("🚀 Benchmark Başlat", type="primary", use_container_width=True):
        with st.status("Benchmark çalışıyor...", expanded=True) as status:
            result = run_benchmark_from_dashboard(model_name, retriever_name, prompt_v, chunk_v, top_k)

            if result.returncode == 0:
                status.write("Benchmark tamamlandı.")
                if result.stdout:
                    st.code(result.stdout[:7000])
                status.update(label="Başarılı", state="complete", expanded=False)
                st.cache_data.clear()
                st.success("Benchmark tamamlandı. Sayfayı yenile veya menüler arasında gez.")
            else:
                status.update(label="Hata oluştu", state="error", expanded=True)
                st.error(result.stderr if result.stderr else "Bilinmeyen hata")

# =========================================================
# 3. DATASET MANAGER
# =========================================================

elif page == "🗂️ Dataset Manager":
    st.subheader("Dataset Manager")

    # Güvenli veri yükleme
    dataset_json = load_dataset() or []
    corpus_json = load_corpus() or []

    # filtered_df yoksa boş dataframe kullan
    try:
        metric_row(filtered_df if not filtered_df.empty else pd.DataFrame())
    except Exception:
        metric_row(pd.DataFrame())

    st.markdown("### Soru Seti Üretimi")

    mode = st.radio(
        "Mod Seç",
        ["Manuel Oluşturma", "Yarı Otomatik Üretim", "İnsan Onayı / Review"],
        horizontal=True
    )

    CATEGORY_OPTIONS = [
        "Faktüel",
        "Çapraz Referans",
        "Tablo/Şekil Yorumu",
        "Yorum/Çıkarım",
        "Kapsam Dışı Kontrol"
    ]
    DIFFICULTY_OPTIONS = ["easy", "medium", "hard"]

    # -----------------------------------------------------
    # 1) MANUEL OLUŞTURMA
    # -----------------------------------------------------
    if mode == "Manuel Oluşturma":
        st.markdown("#### Manuel Soru Ekleme")

        doc_ids = get_doc_ids_from_corpus(corpus_json)

        if not doc_ids:
            st.warning("Corpus içinde doc_id bulunamadı.")
        else:
            with st.form("manual_question_form"):
                doc_id = st.selectbox("Doc ID", doc_ids)
                question = st.text_area("Soru", height=100)
                gold_answer = st.text_area("Altın Cevap (Ground Truth)", height=100)
                source_passage = st.text_area("İlgili Kaynak Pasaj", height=140)

                col1, col2, col3 = st.columns(3)
                with col1:
                    category = st.selectbox("Kategori", CATEGORY_OPTIONS)
                with col2:
                    difficulty = st.selectbox("Difficulty", DIFFICULTY_OPTIONS)
                with col3:
                    answerable = st.selectbox("Answerable", [True, False], index=0)

                approved = st.checkbox("Onaylı olarak kaydet", value=True)

                chunk_candidates = get_chunks_by_doc_id(corpus_json, doc_id) or []
                chunk_options = []
                chunk_map = {}

                for ch in chunk_candidates[:100]:
                    chunk_id = str(ch.get("chunk_id", ""))
                    preview = str(ch.get("text", ""))[:120].replace("\n", " ")
                    label = f"{chunk_id} | {preview}"
                    chunk_options.append(label)
                    chunk_map[label] = chunk_id

                selected_chunk_labels = st.multiselect(
                    "Gold Chunk ID(ler) seç",
                    options=chunk_options
                )

                submitted = st.form_submit_button("➕ Soruyu Kaydet", use_container_width=True)

                if submitted:
                    if not question.strip():
                        st.error("Soru boş olamaz.")
                    elif not gold_answer.strip():
                        st.error("Altın cevap boş olamaz.")
                    elif not source_passage.strip():
                        st.error("Kaynak pasaj boş olamaz.")
                    else:
                        gold_chunk_ids = [
                            chunk_map[x] for x in selected_chunk_labels if x in chunk_map
                        ]

                        try:
                            new_item = add_manual_question(
                                dataset=dataset_json,
                                question=question.strip(),
                                gold_answer=gold_answer.strip(),
                                source_passage=source_passage.strip(),
                                category=category,
                                difficulty=difficulty,
                                answerable=answerable,
                                approved=approved,
                                doc_id=doc_id,
                                gold_chunk_ids=gold_chunk_ids,
                            )
                            st.success(f"Soru kaydedildi: {new_item['id']}")
                            st.json(new_item)
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Soru kaydedilirken hata oluştu: {e}")

    # -----------------------------------------------------
    # 2) YARI OTOMATİK ÜRETİM
    # -----------------------------------------------------
    elif mode == "Yarı Otomatik Üretim":
        st.markdown("#### LLM ile Candidate Soru Üretimi")
        st.caption("Otomatik oluşturulan sorular doğrudan benchmark'a alınmaz. Önce approved=False olarak kaydedilir.")

        doc_ids = get_doc_ids_from_corpus(corpus_json)

        if not doc_ids:
            st.warning("Corpus içinde doc_id bulunamadı.")
        else:
            selected_doc_id = st.selectbox("Doc ID seç", doc_ids)
            chunks = get_chunks_by_doc_id(corpus_json, selected_doc_id) or []

            if not chunks:
                st.warning("Bu doc_id için chunk bulunamadı.")
            else:
                st.markdown(f"**Toplam chunk:** {len(chunks)}")

                chunk_preview_count = st.slider(
                    "Önizlemede gösterilecek chunk sayısı",
                    min_value=1,
                    max_value=min(20, len(chunks)),
                    value=min(5, len(chunks))
                )

                for i, ch in enumerate(chunks[:chunk_preview_count], start=1):
                    with st.expander(f"Chunk {i} | {ch.get('chunk_id', '')}", expanded=False):
                        st.write(ch.get("text", ""))

                col1, col2, col3 = st.columns(3)
                with col1:
                    n_questions = st.slider("Kaç aday soru üretilecek?", 1, 10, 5)
                with col2:
                    category = st.selectbox("Kategori", CATEGORY_OPTIONS, key="semi_category")
                with col3:
                    difficulty = st.selectbox("Difficulty", DIFFICULTY_OPTIONS, key="semi_difficulty")

                session_key = f"generated_candidates_{selected_doc_id}"

                if st.button("🤖 Candidate Soruları Üret", use_container_width=True):
                    try:
                        candidates = simple_candidate_generation_from_chunks(
                            chunks,
                            n_questions=n_questions
                        )
                        if not candidates:
                            st.error("Candidate üretilemedi.")
                        else:
                            st.session_state[session_key] = candidates
                            st.success(f"{len(candidates)} candidate soru üretildi.")
                    except Exception as e:
                        st.error(f"Candidate üretiminde hata oluştu: {e}")

                candidates = st.session_state.get(session_key, [])

                if candidates:
                    st.markdown("### Candidate Sorular")
                    edited_candidates = []

                    for idx, cand in enumerate(candidates, start=1):
                        with st.expander(f"Aday Soru {idx}", expanded=True):
                            q = st.text_area(
                                f"Soru {idx}",
                                value=cand.get("question", ""),
                                key=f"{session_key}_q_{idx}"
                            )
                            ga = st.text_area(
                                f"Gold Answer {idx}",
                                value=cand.get("gold_answer", ""),
                                key=f"{session_key}_a_{idx}"
                            )
                            sp = st.text_area(
                                f"Source Passage {idx}",
                                value=cand.get("source_passage", ""),
                                key=f"{session_key}_s_{idx}"
                            )
                            keep = st.checkbox(
                                f"Bu adayı kaydet",
                                value=True,
                                key=f"{session_key}_keep_{idx}"
                            )

                            if keep and q.strip() and ga.strip() and sp.strip():
                                edited_candidates.append({
                                    "question": q.strip(),
                                    "gold_answer": ga.strip(),
                                    "source_passage": sp.strip(),
                                    "gold_chunk_ids": cand.get("gold_chunk_ids", [])
                                })

                    if st.button("💾 Candidate Soruları Review Queue'ya Kaydet", use_container_width=True):
                        if not edited_candidates:
                            st.warning("Kaydedilecek aday soru yok.")
                        else:
                            try:
                                created = add_candidate_questions(
                                    dataset=dataset_json,
                                    candidates=edited_candidates,
                                    doc_id=selected_doc_id,
                                    category=category,
                                    difficulty=difficulty,
                                )
                                st.success(f"{len(created)} candidate soru approved=False olarak kaydedildi.")
                                st.cache_data.clear()
                                st.session_state.pop(session_key, None)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Candidate kayıt hatası: {e}")

    # -----------------------------------------------------
    # 3) İNSAN ONAYI / REVIEW
    # -----------------------------------------------------
    elif mode == "İnsan Onayı / Review":
        st.markdown("#### Review Queue")

        ds_df = pd.DataFrame(dataset_json) if dataset_json else pd.DataFrame()

        if ds_df.empty:
            st.info("Dataset boş.")
        else:
            # Güvenli filtreleme
            if "created_by" not in ds_df.columns:
                ds_df["created_by"] = ""
            if "approved" not in ds_df.columns:
                ds_df["approved"] = False

            review_df = ds_df[
                (ds_df["created_by"].astype(str) == "llm_candidate") &
                (ds_df["approved"].fillna(False).astype(bool) == False)
            ].copy()

            if review_df.empty:
                st.success("Onay bekleyen candidate soru yok.")
            else:
                st.write(f"Onay bekleyen soru sayısı: {len(review_df)}")

                review_ids = review_df["id"].astype(str).tolist()
                selected_id = st.selectbox("İncelenecek soru", review_ids)

                item_index = None
                item = None

                for i, d in enumerate(dataset_json):
                    if str(d.get("id", "")) == str(selected_id):
                        item_index = i
                        item = d
                        break

                if item is not None:
                    current_category = item.get("category", "Faktüel")
                    if current_category not in CATEGORY_OPTIONS:
                        current_category = "Faktüel"

                    current_difficulty = item.get("difficulty", "easy")
                    if current_difficulty not in DIFFICULTY_OPTIONS:
                        current_difficulty = "easy"

                    current_answerable = bool(item.get("answerable", True))

                    question = st.text_area("Soru", item.get("question", ""), key="review_question")
                    gold_answer = st.text_area("Gold Answer", item.get("gold_answer", ""), key="review_answer")
                    source_passage = st.text_area("Source Passage", item.get("source_passage", ""), key="review_source")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        category = st.selectbox(
                            "Kategori",
                            CATEGORY_OPTIONS,
                            index=CATEGORY_OPTIONS.index(current_category),
                            key="review_category"
                        )
                    with col2:
                        difficulty = st.selectbox(
                            "Difficulty",
                            DIFFICULTY_OPTIONS,
                            index=DIFFICULTY_OPTIONS.index(current_difficulty),
                            key="review_difficulty"
                        )
                    with col3:
                        answerable = st.selectbox(
                            "Answerable",
                            [True, False],
                            index=0 if current_answerable else 1,
                            key="review_answerable"
                        )

                    c1, c2 = st.columns(2)

                    with c1:
                        if st.button("✅ Onayla ve Benchmark Setine Dahil Et", use_container_width=True):
                            try:
                                dataset_json[item_index]["question"] = question.strip()
                                dataset_json[item_index]["gold_answer"] = gold_answer.strip()
                                dataset_json[item_index]["source_passage"] = source_passage.strip()
                                dataset_json[item_index]["category"] = category
                                dataset_json[item_index]["difficulty"] = difficulty
                                dataset_json[item_index]["answerable"] = answerable
                                dataset_json[item_index]["approved"] = True

                                save_dataset(dataset_json)
                                st.success(f"{selected_id} onaylandı.")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Onaylama sırasında hata oluştu: {e}")

                    with c2:
                        if st.button("🗑️ Reddet / Sil", use_container_width=True):
                            try:
                                dataset_json.pop(item_index)
                                save_dataset(dataset_json)
                                st.warning(f"{selected_id} silindi.")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Silme sırasında hata oluştu: {e}")

# =========================================================
# 4. LEADERBOARD
# =========================================================
elif page == "🏆 Leaderboard":
    st.subheader("Leaderboard")
    if filtered_df.empty:
        st.warning("Gösterilecek sonuç yok.")
    else:
        metric_row(filtered_df)

        sort_option = st.selectbox(
            "Sıralama seç",
            [
                "F1’e göre",
                "Recall’a göre",
                "Düşük hallucination’a göre",
                "Düşük latency’ye göre",
            ],
            index=0
        )

        sort_map = {
            "F1’e göre": ("f1", False),
            "Recall’a göre": ("recall", False),
            "Düşük hallucination’a göre": ("hallucination", True),
            "Düşük latency’ye göre": ("latency", True),
        }

        sort_col, asc = sort_map[sort_option]

        cols = [c for c in [
            "run_id", "model", "retriever", "prompt_v", "chunk_v",
            "f1", "recall", "precision", "hallucination", "latency", "tarih", "timestamp"
        ] if c in filtered_df.columns]

        leaderboard_df = filtered_df[cols].copy()
        leaderboard_df = leaderboard_df.sort_values(by=sort_col, ascending=asc)
        st.dataframe(leaderboard_df, use_container_width=True, height=550)

# =========================================================
# 5. RUN COMPARISON
# =========================================================
elif page == "🆚 Run Comparison":
    st.subheader("Run Comparison")
    if run_summary_df.empty or run_summary_df["run_id"].nunique() < 2:
        st.warning("Karşılaştırma için en az 2 run gerekli.")
    else:
        metric_row(filtered_df if not filtered_df.empty else results_df)

        run_ids = run_summary_df["run_id"].dropna().astype(str).unique().tolist()

        c1, c2 = st.columns(2)
        run_a = c1.selectbox("Run A", run_ids, index=0)
        run_b = c2.selectbox("Run B", run_ids, index=min(1, len(run_ids) - 1))

        compare_df = run_summary_df[run_summary_df["run_id"].astype(str).isin([run_a, run_b])]

        cols = [c for c in [
            "run_id", "model", "retriever", "prompt_v", "chunk_v", "top_k",
            "f1", "exact_match", "recall", "precision", "mrr", "hit_at_k",
            "latency", "hallucination"
        ] if c in compare_df.columns]
        st.dataframe(compare_df[cols], use_container_width=True)

        st.markdown("### Metric Bazlı Karşılaştırma")
        metric_cols = [c for c in [
            "f1", "exact_match", "recall", "precision", "mrr", "hit_at_k", "latency", "hallucination"
        ] if c in compare_df.columns]

        fig = px.bar(compare_df, x="run_id", y=metric_cols, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 6. ERROR ANALYSIS
# =========================================================
elif page == "❌ Error Analysis":
    st.subheader("Error Analysis")
    if filtered_df.empty:
        st.warning("Analiz için veri yok.")
    else:
        metric_row(filtered_df)

        mode = st.radio(
            "Filtre",
            ["status = error", "f1 < threshold", "exact_match = 0", "hallucination yüksek"],
            horizontal=True
        )

        work_df = filtered_df.copy()

        if mode == "status = error" and "status" in work_df.columns:
            work_df = work_df[work_df["status"].astype(str) == "error"]

        elif mode == "f1 < threshold" and "f1" in work_df.columns:
            threshold = st.slider("F1 threshold", 0.0, 1.0, 0.5, 0.05)
            work_df = work_df[work_df["f1"] < threshold]

        elif mode == "exact_match = 0" and "exact_match" in work_df.columns:
            work_df = work_df[work_df["exact_match"] == 0]

        elif mode == "hallucination yüksek" and "hallucination" in work_df.columns:
            threshold = st.slider("Hallucination threshold", 0.0, 1.0, 0.5, 0.05)
            work_df = work_df[work_df["hallucination"] > threshold]

        cols = [c for c in [
            "question", "ground_truth", "generated_answer", "retrieved_context",
            "error_message", "category", "retriever", "model", "f1",
            "exact_match", "hallucination", "status"
        ] if c in work_df.columns]

        st.dataframe(work_df[cols], use_container_width=True, height=550)

# =========================================================
# 7. QUESTION TRACE
# =========================================================
elif page == "🧩 Question Trace":
    st.subheader("Question Trace")
    if filtered_df.empty or "question_id" not in filtered_df.columns:
        st.warning("Question trace için veri yok.")
    else:
        metric_row(filtered_df)

        qids = filtered_df["question_id"].dropna().astype(str).unique().tolist()
        selected_qid = st.selectbox("Question ID", sorted(qids))

        qdf = filtered_df[filtered_df["question_id"].astype(str) == selected_qid]
        row = qdf.iloc[-1]

        st.markdown("### Question")
        st.write(row.get("question", ""))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Gold Answer")
            st.write(row.get("ground_truth", ""))

        with c2:
            st.markdown("### Generated Answer")
            st.write(row.get("generated_answer", ""))

        st.markdown("### Retrieval")
        gold_chunk_ids = parse_json_list(row.get("gold_chunk_ids", "[]"))
        retrieved_chunk_ids = parse_json_list(row.get("retrieved_chunk_ids", "[]"))
        retrieved_context = parse_json_list(row.get("retrieved_context", "[]"))

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Gold Chunk IDs**")
            st.code("\n".join(map(str, gold_chunk_ids)) if gold_chunk_ids else "Yok")
        with c4:
            st.markdown("**Retrieved Chunk IDs**")
            st.code("\n".join(map(str, retrieved_chunk_ids)) if retrieved_chunk_ids else "Yok")

        st.markdown("### Retrieved Context")
        if retrieved_context:
            for i, chunk in enumerate(retrieved_context, start=1):
                with st.expander(f"Chunk {i}", expanded=(i == 1)):
                    st.write(chunk)
        else:
            st.info("Retrieved context bulunamadı.")

        st.markdown("### Metrics")
        payload = {
            k: row.get(k, None)
            for k in [
                "f1", "exact_match", "recall", "precision", "mrr",
                "hit_at_k", "hallucination", "faithfulness",
                "latency", "status", "run_id", "model", "retriever",
                "prompt_v", "chunk_v", "top_k"
            ] if k in row.index
        }
        st.json(payload)

# =========================================================
# 8. DATASET ANALYTICS
# =========================================================
elif page == "📚 Dataset Analytics":
    st.subheader("Dataset Analytics")
    metric_row(filtered_df if not filtered_df.empty else pd.DataFrame())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Toplam Soru", health["total"])
    c2.metric("Answerable", health["answerable_yes"])
    c3.metric("Unanswerable", health["answerable_no"])
    c4.metric("Onaylı", health["approved_yes"])
    c5.metric("Onaysız", health["approved_no"])

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("### Kategori Dağılımı")
        cat_df = pd.DataFrame(
            [{"category": k, "count": v} for k, v in health["categories"].items()]
        )
        if not cat_df.empty:
            fig = px.bar(cat_df, x="category", y="count", color="category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kategori verisi yok.")

    with right:
        st.markdown("### Difficulty Dağılımı")
        diff_df = pd.DataFrame(
            [{"difficulty": k, "count": v} for k, v in health["difficulty_counts"].items()]
        )
        if not diff_df.empty:
            fig = px.bar(diff_df, x="difficulty", y="count", color="difficulty")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Difficulty verisi yok.")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### PDF Başına Soru Sayısı")
        pdf_df = pd.DataFrame(
            [{"doc_id": k, "count": v} for k, v in health["doc_counts"].items()]
        )
        if not pdf_df.empty:
            fig = px.bar(pdf_df, x="doc_id", y="count", color="doc_id")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("doc_id verisi yok.")

    with c4:
        st.markdown("### Answerable / Unanswerable Oranı")
        ans_df = pd.DataFrame({
            "type": ["Answerable", "Unanswerable"],
            "count": [health["answerable_yes"], health["answerable_no"]]
        })
        fig = px.pie(ans_df, names="type", values="count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Dataset Preview")
    if dataset_json:
        st.dataframe(pd.DataFrame(dataset_json), use_container_width=True, height=450)
    else:
        st.info("dataset.json boş.")

# =========================================================
# 9. EXPORT
# =========================================================
elif page == "💾 Export":
    st.subheader("Export")
    if filtered_df.empty:
        st.warning("Dışa aktarılacak veri yok.")
    else:
        metric_row(filtered_df)

        preview_cols = [c for c in [
            "run_id", "question_id", "question", "generated_answer",
            "ground_truth", "f1", "exact_match", "recall", "precision",
            "hallucination", "latency", "status"
        ] if c in filtered_df.columns]

        st.dataframe(filtered_df[preview_cols], use_container_width=True, height=500)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 CSV indir",
                data=export_bytes(filtered_df, "csv"),
                file_name=f"rag_benchmark_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "📥 JSON indir",
                data=export_bytes(filtered_df, "json"),
                file_name=f"rag_benchmark_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )