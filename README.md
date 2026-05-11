# Rag_Benchmark_Project
# RAG Benchmark & Evaluation System

<p align="center">
A framework to evaluate Retrieval-Augmented Generation (RAG) systems with benchmark datasets, metrics, and an interactive dashboard.
</p>

<img width="1909" height="915" alt="Ekran görüntüsü 2026-03-13 104456" src="https://github.com/user-attachments/assets/a05a6dc6-9870-4661-a042-83ae3af823ac" />

---


## Overview

This project provides a **complete benchmarking framework for RAG systems**.

It allows you to:

- run benchmark tests on RAG pipelines
- evaluate retrieval and generation performance
- measure hallucination rates
- compare models and retrievers
- visualize results in a Streamlit dashboard

The system is designed for **research, experimentation, and evaluation of LLM-based retrieval pipelines**.

---

## Features

- RAG pipeline evaluation
- Benchmark dataset support
- Retrieval metrics
- Generation metrics
- Hallucination detection
- Model comparison
- Streamlit dashboard
- Dataset management interface

---

## Project Structure

```
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
└── requirements.txt
```

---

## System Architecture

```
Question
   │
   ▼
Retriever
   │
   ▼
Relevant Document Chunks
   │
   ▼
Generator (LLM)
   │
   ▼
Generated Answer
   │
   ▼
Evaluation Metrics
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rag-benchmark.git
cd rag-benchmark
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows

```
.venv\Scripts\activate
```

Mac/Linux

```
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is not available:

```bash
pip install streamlit pandas numpy sentence-transformers scikit-learn google-generativeai chromadb python-dotenv plotly
```

---

## API Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

---

## Running Benchmark

Run the benchmark evaluation:

```bash
python -m benchmark.evaluate
```

The benchmark will:

1. load questions from the dataset
2. retrieve relevant document chunks
3. generate answers with the LLM
4. compute evaluation metrics
5. store results for analysis

---

## Running the Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard allows you to:

- manage datasets
- inspect benchmark runs
- compare models
- visualize metrics
- analyze hallucination rates

---

## Evaluation Metrics

The system evaluates RAG pipelines using several metrics.

### Retrieval Metrics

- Recall
- Precision
- Context Recall

### Generation Metrics

- Exact Match
- F1 Score
- Semantic Similarity

### Hallucination Metrics

The system measures how often the model generates answers that **are not supported by the retrieved context**.

---

## Example Hallucination Test

Example unanswerable question:

```
What is the annual budget of TÜBİTAK MAM?
```

Expected behavior:

```
The information is not present in the provided documents.
```

If the model generates an unsupported answer, it is counted as **hallucination**.

---

## How to run the project

1. Clone th repository
  >git clone <repo-link>
  >cd Rag_project

2. Create & Activate Virtual Environment
  > python -m venv .venv
  >.\.venv\Scripts\Activate


3. Install Dependencies 

 > pip install pandas numpy streamlit plotly scikit-learn sentence-transformers python-dotenv openai google-generativeai groq

4. Setup Environment Variables

>OPENAI_API_KEY=your_key_here
>GOOGLE_API_KEY=your_key_here
>GROQ_API_KEY=your_key_here

5.Run Benchmark
>python -m benchmark.evaluate

6.Dashboard
>streamlit run dashboard/app.py



## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- ChromaDB
- Gemini API
- Pandas
- Plotly

---

## Future Improvements

- advanced evaluation metrics
- automatic dataset generation
- model leaderboard
- multi-model benchmark support
- RAG pipeline visualization

---
