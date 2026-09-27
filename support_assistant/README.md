# Zepto GenAI Support Assistant

A small but complete GenAI (RAG) service for Zepto's policy corpus.

- **Retrieval** is real in both modes (local embeddings + ChromaDB, no API key).
- **Generation** branches on a single environment variable, `MOCK_LLM`.
- **Graded baseline** runs fully offline with `MOCK_LLM` unset or `MOCK_LLM=1`
  — no signup, no API key, no network call to any LLM provider.
- **Real-LLM mode** (`MOCK_LLM=0`) is an optional, ungraded extension.

---

## Files

```
support_assistant/
├── docs/
│   ├── doc_01.txt   # Delivery Policy
│   ├── doc_02.txt   # Returns & Refunds
│   ├── doc_03.txt   # Membership Tiers
│   ├── doc_04.txt   # Order Tracking
│   ├── doc_05.txt   # Order Cancellation Policy
│   ├── doc_06.txt   # Damaged or Missing Items
│   ├── doc_07.txt   # Gift Cards
│   └── doc_08.txt   # Customer Support Hours
├── ingest.py                # load → chunk → embed → ChromaDB
├── main.py                  # LangGraph StateGraph + FastAPI app
├── requirements.txt
├── Dockerfile
├── chroma_db/               # created by ingest.py
└── README.md
```

---

## Requirements

```
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.6
langgraph>=0.2
chromadb>=0.5
sentence-transformers>=2.7
requests>=2.31
```

Use **Python 3.10 or 3.11**. Python 3.12 has wheel-availability problems for
`chromadb` / `torch` / `onnxruntime`.

---

## Setup & run (graded mock baseline)

```bash
cd support_assistant

# 1. Create and activate a venv
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies (CPU-only torch keeps it smaller)
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# 3. Build the ChromaDB index (offline embeddings)
python ingest.py                     # prints: "indexed NN chunks across 8 docs"

# 4. Start the FastAPI server with MOCK_LLM at its default
# Windows PowerShell
$env:MOCK_LLM="1"
# Windows cmd
set MOCK_LLM=1
# macOS / Linux
export MOCK_LLM=1

uvicorn main:app --host 0.0.0.0 --port 7860
```

The first run of `python ingest.py` downloads `all-MiniLM-L6-v2` (~90 MB) from
Hugging Face and caches it locally. After that, everything runs offline.

---

## Example calls (recorded with `MOCK_LLM` at its default)

Run these from a **second** terminal while the server is up.

### 1) Policy question → routed to `retrieve_and_answer`

```bash
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the delivery fee for orders under INR 149?"}'
```

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders ove",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

### 2) General question → routed to `direct_answer`

```bash
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Who won the 2018 FIFA World Cup?"}'
```

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

On Windows PowerShell, use `Invoke-RestMethod` instead of `curl`:

```powershell
Invoke-RestMethod -Uri http://localhost:7860/ask -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the delivery fee for orders under INR 149?"}'
```

FastAPI's interactive docs are available at
`http://localhost:7860/docs`.

---

## Docker

Build and run locally (no push required):

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
# then POST http://localhost:7860/ask
```

The `Dockerfile`:

- installs Python dependencies,
- copies `docs/` and the two source files,
- runs `python ingest.py` at build time (offline embeddings),
- sets `MOCK_LLM=1` by default,
- serves `/ask` via uvicorn on port 7860.

---

## Pipeline architecture

The RAG pipeline has four stages, all real in both modes except generation,
which branches on `MOCK_LLM`.

### 1. Ingestion

`ingest.py` reads all eight `docs/doc_*.txt` files and splits each into
fixed-size chunks of 300 characters with 50 characters of overlap
(`chunk()` helper). Each chunk gets an ID of the form
`<doc_id>::chunk<index>` and metadata `{"doc_id": ..., "chunk_index": ...}`.

### 2. Embedding

The same `ingest.py` wraps `all-MiniLM-L6-v2` through ChromaDB's
`SentenceTransformerEmbeddingFunction` (local, no API, no network after the
first download) and persists the vectors into a ChromaDB collection named
`zepto_policies` under `./chroma_db/`.

### 3. Retrieval

In `main.py`, the LangGraph node `retrieve_and_answer` embeds the query with
the same MiniLM model and asks ChromaDB for the top-3 nearest chunks. **This
stage always runs for real, in both mock and real-LLM modes** — it needs no
API key and is unaffected by `MOCK_LLM`.

### 4. Generation

The same `retrieve_and_answer` node produces the final answer. This is the
**only stage that branches on `MOCK_LLM`**:

- **`MOCK_LLM=1` (default, graded)**: no LLM call. Returns the canned string
  `"Based on the retrieved context: <first 200 chars of top chunk>"`, sets
  `sources = [top_doc_id]`, and `confidence = 1.0`.
- **`MOCK_LLM=0` (optional, ungraded)**: builds the structured
  `PROMPT_TEMPLATE` and calls the LLM. The response is validated against the
  `AskResponse` Pydantic model; on failure, the call is retried up to 2
  additional times with a corrective instruction before returning an error
  response.

### Routing

`classify_intent` uses a keyword heuristic when `MOCK_LLM=1`:

```python
KEYWORDS = ["delivery", "return", "refund", "membership",
            "tracking", "cancel", "gift card", "support hours"]
```

If the lowercased query contains any keyword → `policy_question`, else →
`general_question`. When `MOCK_LLM=0`, this step calls the LLM instead. A
conditional edge (`route`) sends the state to `retrieve_and_answer` for
`policy_question` or to `direct_answer` for `general_question`.

```
docs/*.txt
    │  ingest.py — chunk + embed with all-MiniLM-L6-v2
    ▼
chroma_db  (ChromaDB collection: zepto_policies)
    ▲
    │  main.py — retrieve_and_answer (retrieval always real)
classify_intent ──route──► retrieve_and_answer ─► AskResponse
    └───route──► direct_answer ─────────────────► AskResponse
```

**Where `MOCK_LLM` branches**

| Stage | Component | Behaviour |
|-------|-----------|-----------|
| Ingestion | `ingest.py` | identical in both modes |
| Embedding | `ingest.py` (MiniLM) | identical in both modes |
| Retrieval | `main.py` → `retrieve_and_answer` | identical in both modes (always real) |
| Classification | `main.py` → `classify_intent` | keyword heuristic in mock; LLM call in real |
| Answer generation | `main.py` → `retrieve_and_answer` | canned template in mock; LLM + JSON validation in real |
| Direct answer | `main.py` → `direct_answer` | fixed string in mock; LLM call in real |

---

## Structured prompt template (used only when `MOCK_LLM=0`)

The full template lives at `PROMPT_TEMPLATE` in `main.py` and contains all the
required skeleton components:

- **Role** — "You are Zepto's customer-support assistant…"
- **Context** — retrieved chunks with their doc IDs
- **Task** — answer using only the context
- **Negative constraint** — "Do not answer using information not present in
  the provided context."
- **Few-shot example** — the Zepto Pass cost Q/A
- **Format** — strict JSON `{answer, sources, confidence}`
- **Length** — 1–3 sentences

---

## Output schema

Every `/ask` response conforms to this Pydantic model:

```python
class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []
    confidence: float = 1.0
```

In mock mode, `sources` is populated deterministically from the retrieved
chunk's `doc_id` (top chunk only), and `confidence` is fixed at `1.0`. In
real-LLM mode, the raw LLM output is parsed and validated against this model;
on failure the model is re-prompted up to 2 more times before returning a
clearly marked error response.

---

## Optional extensions (ungraded)

- **Real LLM (Groq free tier)**: sign up at <https://console.groq.com>
  (free tier — no credit card). Then run with:
  ```bash
  export MOCK_LLM=0
  export GROQ_API_KEY=your_key_here
  uvicorn main:app --host 0.0.0.0 --port 7860
  ```
  Never commit the API key to the repository.
- **Hugging Face Spaces deployment**: not attempted. Not required for
  grading.

These extensions do not affect the graded baseline.

---

## Deliverables checklist

- [x] All 8 corpus documents chunked, embedded, and queryable from ChromaDB
- [x] Structured prompt template shows all 5 skeleton components (role,
      context, task, format, length) + negative constraint + few-shot example,
      as actual text in `main.py`
- [x] With `MOCK_LLM` at its default, `classify_intent`'s keyword heuristic
      routes at least one policy-style query to `policy_question` and at least
      one unrelated query to `general_question`, with no LLM call
- [x] LangGraph StateGraph has the 3 named nodes and a working conditional
      edge, with an example of each route run in mock mode
- [x] Retrieval returns chunks from the correct source document (real in both
      modes)
- [x] `retrieve_and_answer`'s mock output follows the
      `"Based on the retrieved context: …"` template; `direct_answer`'s mock
      output is a fixed canned string; neither makes a network call when
      `MOCK_LLM` is at its default
- [x] `AskResponse` schema (`answer` / `sources` / `confidence`) is populated
      deterministically in mock mode; retry-on-failure logic is present in
      code for the optional real-LLM path
- [x] FastAPI app runs locally via uvicorn; both example JSON responses are
      shown above
- [x] `Dockerfile` is present, correctly configured, and documented as
      locally buildable / runnable
- [x] README architecture describes ingestion → embedding → retrieval →
      generation, names the component handling each stage, and states where
      `MOCK_LLM` branches

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'chromadb'` | Run `python -m pip install -r requirements.txt` **inside the venv**. Verify with `python -c "import sys; print(sys.executable)"` — the path must contain `.venv`. |
| `Collection zepto_policies does not exist` | Run `python ingest.py` first. |
| `No suitable Python runtime found` | Install Python 3.11 from python.org; 3.12 is not recommended for this module. |
| `Activate.ps1 cannot be loaded…` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then re-activate. |
| Port 7860 already in use | `uvicorn main:app --port 7861`. |
| First `python ingest.py` hangs | It is downloading `all-MiniLM-L6-v2` (~90 MB). Subsequent runs use the cache. |
