# 📚 RagTimeAPI

**RagTimeAPI** is a self-hosted REST API designed to power Retrieval-Augmented Generation (RAG) workflows with LLMs. It delivers relevant context based on document similarity to enhance prompt quality.

---

## 🚀 Features

- 🔍 Sentence similarity scoring with `CrossEncoder` or `BiEncoder`
- 📄 Document chunking + filtering
- 🧠 Integrate with any LLM (Gemini, OpenAI, local, etc.)
- 🧪 Tested with [Vault](https://github.com/AliBenrami/vault)

---

## 🧑‍💻 Getting Started

### Prerequisites

- Python ≥ 3.9
- FastAPI
- sentence-transformers
- (Optional) vector DB or file-based document store

### Installation

```bash
git clone https://github.com/AliBenrami/RagTimeAPI.git
cd RagTimeAPI
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🔧 Endpoints

### `POST /call-rag-llm`

```json
{
  "prompt": "What is quantum entanglement?",
  "history": [],
  "model": "gemini-2.0-flash"
}
```

Returns relevant context chunks and an LLM response based on the retrieved context.

---

## 🔩 Internals

- Uses sentence-transformers for scoring
- Simple cosine similarity or cross-encoder reranking
- Can be plugged into any chat frontend or bot system

---

## 📄 License

MIT
